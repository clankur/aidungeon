- [ ] storyteller/dungeon master to give encompassing narrative
  - [ ] build knowledge graph based off of initial narrative?
- [ ] adjust to user feedback

- [ ] event log
  - event

    ```json
        {
            action: "shot",
            target: "entity_id"
            timestamp: 2024-10-12 12:00:00:00
        }
    ```

- [ ] maintain per character event history (?)

- [ ] have an hierachal story from storyteller
  - [ ] plotbeats

- relationship graph
  - [ ] ground interactions between entities and player
  - [ ] each entity consists of:

    ```json
    {
        name: "jimmy"
        entity_id: "guid"
        state: dead
        inventory: []
    }
    ```

- [ ] dfs to update relationships after an event
- [ ] topsort can be good for tracking information propagation

- seperate the variables
- [ ] ai dungeon
  - minimize hallucinations
  - w/ relationship graph
- [ ] sims like game
