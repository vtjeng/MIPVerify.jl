using Memento

function getlogger()::Memento.Logger
    Memento.getlogger(current_module())
end

function setloglevel!(level::String)
    # Options correspond to Memento.jl's levels.
    # https://invenia.github.io/Memento.jl/latest/man/intro.html#Logging-levels-1
    logger = MIPVerify.getlogger()
    Memento.setlevel!(logger, level)
    while logger.name != "root"
        logger = Memento.getparent(logger.name)
        Memento.setlevel!(logger, level)
    end
end

