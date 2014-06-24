package = "eladtools"
version = "scm-1"

source = {
   url = "eladtools.torch",
   tag = "master"
}

description = {
   summary = "Torch Tools Made By Elad Hoffer",
   detailed = [[
   	    Tools for torch packages
   ]],
   homepage = "https://github.com/ehoffer/eladtools.torch"
}

dependencies = {
  
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}
