#!/usr/bin/env lua
-- compile.lua
-- @author Xein

-- Detect platform
local is_windows = package.config:sub(1,1) == "\\"

local function detect_unix_name()
  local p = io.popen("uname -s 2>/dev/null")
  if not p then return "" end
  local s = p:read("*l") or ""
  p:close()
  return s:lower()
end

local unix_name = is_windows and "" or detect_unix_name()

-- Build config
local build_dir = "./bin/debug"
local build_type = "Debug"

-- Generator and compilers by platform
local generator
local c_compiler
local cpp_compiler

if is_windows then
  generator = "Visual Studio 17 2022"
else
  generator = "Unix Makefiles"
  if unix_name:find("darwin") or unix_name:find("mac") then
    c_compiler = "clang"
    cpp_compiler = "clang++"
  else
    c_compiler = "gcc"
    cpp_compiler = "g++"
  end
end

local compile_commands_src = build_dir .. "/compile_commands.json"
local compile_commands_dst = "./compile_commands.json"

-- Function to run a command
local function run_cmd(cmd)
  print("> " .. cmd)
  local ok, exit_type, code = os.execute(cmd)
  if not ok or (exit_type ~= "exit" or code ~= 0) then
    print(string.format("Command failed (type: %s, code: %s)", tostring(exit_type), tostring(code)))
    os.exit(1)
  end
end

-- Small helpers
local function file_exists(path)
  local f = io.open(path, "rb")
  if f then f:close() return true end
  return false
end

local function copy_file(src, dst)
  local in_f = assert(io.open(src, "rb"))
  local data = in_f:read("*a")
  in_f:close()
  local out_f = assert(io.open(dst, "wb"))
  out_f:write(data or "")
  out_f:close()
end

-- Function to copy compile_commands.json (portable)
local function copy_compile_commands()
  print("\n=== Copying compile_commands.json ===")
  if not file_exists(compile_commands_src) then
    print(string.format("No compile_commands.json at %s (skipping).", compile_commands_src))
    return
  end
  copy_file(compile_commands_src, compile_commands_dst)
  print(string.format("Copied to %s", compile_commands_dst))
end

-- Function to run tests
local function run_tests()
  print("\n=== Running tests ===")
  local cmd = string.format("ctest --test-dir %s/tests -VV", build_dir)
  if is_windows then
    cmd = cmd .. string.format(" -C %s", build_type)
  end
  run_cmd(cmd)
end

-- Parse arguments
local args = {}
for i = 1, #arg do
  args[arg[i]] = true
end

local do_generate = args["-g"] or next(args) == nil
local do_build    = args["-b"] or next(args) == nil
local do_test     = args["-t"] or next(args) == nil

-- Generate step
if do_generate then
  -- print("\n=== Downloading submodules ===")
  -- run_cmd("git submodule update --init --recursive")

  print("\n=== Generating build system ===")
  local cmake_parts = {
    'cmake',
    string.format('-S . -B "%s"', build_dir),
    string.format('-G "%s"', generator),
    string.format('-DCMAKE_BUILD_TYPE=%s', build_type),
    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'
  }
  if c_compiler then table.insert(cmake_parts, '-DCMAKE_C_COMPILER=' .. c_compiler) end
  if cpp_compiler then table.insert(cmake_parts, '-DCMAKE_CXX_COMPILER=' .. cpp_compiler) end

  run_cmd(table.concat(cmake_parts, " "))
  copy_compile_commands()
end

-- Build step
if do_build then
  print("\n=== Building project ===")
  local build_cmd = string.format('cmake --build "%s" -j 40', build_dir)
  if is_windows then
    build_cmd = build_cmd .. string.format(' --config %s', build_type)
  end
  run_cmd(build_cmd)
end

-- Test step
-- if do_test then
--   run_tests()
-- end
