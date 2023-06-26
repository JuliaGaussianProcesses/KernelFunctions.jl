module TestUtils

function test_interface end
function test_with_type end
function test_type_stability end
function example_inputs end

function __init__()
    # Better error message if users forget to load Test
    Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
        if exc.f === test_interface ||
            exc.f === test_with_type ||
            exc.f === test_type_stability ||
            exc.f === example_inputs
            print(io, "\\nDid you forget to load Test?")
        end
    end
end

end # module
