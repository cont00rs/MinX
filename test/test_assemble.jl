using Test
using MinX

@testset "Assemble utilities" begin
    # TODO Generalise over more cases.
    source = [1, 2, 3]
    dest = zeros(2*length(source))
    MinX.tile!(dest, source)
    @test (dest[1:3] == source) && (dest[4:end] == source)
    MinX.repeat!(dest, source)
    @test (dest[1:2:end] == source) && (dest[2:2:end] == source)
end
