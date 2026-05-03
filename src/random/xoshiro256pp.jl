function Base.write(file::IO, rng::Xoshiro)
    write(file, rng.s0)
    write(file, rng.s1)
    write(file, rng.s2)
    write(file, rng.s3)
end

function Base.read(file::IO, rng::Xoshiro)
    rng.s0 = read(file, UInt64)
    rng.s1 = read(file, UInt64)
    rng.s2 = read(file, UInt64)
    rng.s3 = read(file, UInt64)
    return rng
end