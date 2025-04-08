# generative agents paper notes

- agent interactions
  - [ ] appears no graph grounding relationships between agents
  - maintain a memory stream = list of memories

  ```json
  memory = {
    description: 'memory description',
    creationTime: TIMESTAMP,
    lastAccessTime: TIMESTAMP,
  }
  ```

  - selectively retrieve memories from memory stream
    - when does this happen? top memories are included within th prompt
      - as a reponse to its current situtation
      - done explicitly in reflection
    - score = recency + importance + relevance

  - [ ] agents perform the following loop to build a memory stream at each time step
    - [ ] observe
      - make a new observation of something perceived by the agent. either
        - an action the agent performed
        - an action percieved by agent that is performed by another agent
      - observation added as a memory
    - [ ] reflect
      - query the latest n records in records in memory stream
        - n = 100
      - prompt model to form k high level questions to reflect on based on records
        - k = 3
      - questions are used to query memory stream for relevant observations
        - retrieve memories from memory stream
      - then prompt LLM to extract insights
        - What are the 5 high level insights can you infer from the above statements? (example format: isight (because 1, 5, 3))
      - output = statement + pointers to the memories cited
      - reflection added as a memory
    - [ ] plan
      - build a high level abstract plan
      - add plan to memory stream
      - recursively decompose plan into finer grained chunks
        - 5-15 min chunks

- [ ] environment represented as a tree data structure
  - [ ] each agent keeps their own local representation of what they know


