# Formula One Race Strategy
## Objective
Find the best tire strategy: When to pit and which compounds to use.
This can depend on Objectives. Overall goal is to finish highest in the standings. What this really means canv vary:
1. Getting the most points -> Finishing in the best place -> Having the fastest overall race time
2. Getting the most points -> Finishing in the best place -> Finishing ahead of the most cars even if race time is slower
3. Maximizing point differential between you and closest rivals -> Finishing as far ahead of them as possible

Current focus: Find optimal strategy to minimize total race time not considering opponent strategy, position, or weather
1. Make functions for estimated lap time with inputs (tire compound, tire age, fuel load)
    - Using gradient boosting
    - Has shown a clear difference in year to year performance
    - Take on the position that youre part way through the season so you can train on previous years race data and estimate
    the difference based on the difference in this years previous race lap times
2. Try a dynamic programming solver

