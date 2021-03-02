begin
    Ds = [2, 3, 4, 5]
    Ws = [25,50,75,100,125,150]

    [HyperParams(; Ws=[W for _ ∈ 1:D]) for W ∈ Ws for D ∈ Ds ]
end
