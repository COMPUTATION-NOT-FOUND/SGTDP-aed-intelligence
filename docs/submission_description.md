I built a planning tool that decides where Singapore should add public-access AEDs, and returns a ranked install queue. It covers Toa Payoh, Ang Mo Kio, Bedok and Jurong East.

The problem: SCDF publishes a list of the AEDs already installed, but a planner with funding for ten more has no way to decide which ten, or which to buy first. A flat list is not a plan.

How it works: I lay an H3 grid over the area and weight each cell by building footprint. Candidates are real OSM buildings that pass an eligibility filter. A greedy algorithm picks sites one at a time by how much new demand each covers, starting from the network already there, so funding ten now and ten later never invalidates the first ten. In Toa Payoh, ten devices take coverage from 47.0% to 56.9%, against 47.7% for the baseline. An ILP checks the result.

What I would improve with more time: a third of the leftover gap has no eligible building within reach, so widening the candidate pool would beat any solver change.
