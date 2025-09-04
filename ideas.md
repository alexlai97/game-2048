idea1:
 now greedy performs consistently to 1024. 
before moving on to ML.
let's try to make the most of it.
How about a slow and fast thinking mode. 
When there's a lot of empty slots on the board, greedy is fine (and below 1024), when after 1024 and few slots, try use Tree search, let's test it.
It's like a combined agent for greedy and tree search, any ideas ? does it make sense? 

idea2:
for greedy, what make it to crash after 1024? advanced greedy
maybe add a some award if it can align itself with a snake like pattern (i know it from playing it from human experience, you also know the trick right?)
let's design the ultimate greedy agent with preferred rules and strategies. 


idea3:
make it a toggle between ai and human
in ai mode, show all different options
for each agent, all per agent settings can be modified on the fly (how about it?) such as search depth (for tree agent), etc
i can take over game, and i can switch between agent in runtime (how about it? what needs to be changed ? think ultra hard. logging ? pros and cons and cost, will it break future design) 
i have to press space button to stop ai. and space again to resume to automatically doing things. 
when i paused the game, i can switch agent. 
(is it possible to switch agents? do they have memory? will switching break their logics and mind ?)


ML:
visualize neural-net