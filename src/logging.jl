using Memento

function set_log_level(level::String)
    # Options correspond to Mememento.jl's log names. 
    Memento.config(level)
end