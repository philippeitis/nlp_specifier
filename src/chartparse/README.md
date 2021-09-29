PERF NOTES:

Using FNVHasher improves overall performance by 25% for enum case
After this, caching hashes improves performance a further 10% 

Performance of these optimizations has not been measured for string case.

SmallVec does not appear to change performance in any meaningful way.
