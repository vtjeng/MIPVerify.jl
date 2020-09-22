using Memento

# Create our module level logger (this will get precompiled)
const LOGGER = getlogger(@__MODULE__)

# Register the module level logger at runtime so that folks can access the logger via
# `getlogger(MyModule)`
# NOTE: If this line is not included then the precompiled `MyModule.LOGGER` won't be
# registered at runtime.
__init__() = Memento.register(LOGGER)

function set_log_level!(level::String)
    # Options correspond to Memento.jl's levels.
    # https://invenia.github.io/Memento.jl/latest/man/intro.html#Logging-levels-1
    Memento.config!(level; recursive = true)
end

set_log_level!("notice")
