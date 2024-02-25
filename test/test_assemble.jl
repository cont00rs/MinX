using Test
using MinX

@testset "Assemble utilities" begin
    len = 10
    source = rand(Int8, len)
    sizes = 1:3

    for n in sizes
        dest = zeros(n * length(source))
        MinX.tile!(dest, source)
        for i = 1:n
            @test dest[1+(i-1)*len:i*len] == source
        end

        dest = zeros(n * length(source))
        MinX.repeat!(dest, source)
        for i = 1:n
            @test dest[i:n:end] == source
        end
    end
end
