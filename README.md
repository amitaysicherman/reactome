**to** get the data for this notebook first use this queries with the [reactome noe4j database](https://reactome.org/dev/graph-database):


input->reaction->output
```
MATCH
    (input_:PhysicalEntity)<-[:input]-(r:ReactionLikeEvent)-[:output]->(output_:PhysicalEntity)
WHERE
            input_.speciesName="Homo sapiens"
        AND
            r.speciesName="Homo sapiens"
        AND
            output_.speciesName="Homo sapiens"
RETURN
    input_.dbId as in,
    r.dbId as r,
    output_.dbId as out
```

reaction->reletion->reaction
```
MATCH
    (r1:ReactionLikeEvent)-[t]->(r2:ReactionLikeEvent)
WHERE
    r1.speciesName="Homo sapiens"
AND
    r2.speciesName="Homo sapiens"
RETURN
    r1.dbId as r1,
    type(t) as t,
    r2.dbId as r2
```

entity -> cell location
```
MATCH (n:PhysicalEntity)-[:compartment]->(m:Compartment)
WHERE
    n.speciesName="Homo sapiens"
RETURN n.dbId as e ,m.dbId as c
```

complex -> entity
```
MATCH (n:Complex)-[:hasComponent]->(m:PhysicalEntity)
WHERE n.speciesName="Homo sapiens"
  AND m.speciesName="Homo sapiens"
RETURN n.dbId as c, m.dbId as e
```

entity -> ref
```
MATCH (n:PhysicalEntity)-[:referenceEntity]->(m)
WHERE n.speciesName="Homo sapiens"
RETURN n.dbId as e ,m.dbId as ref
```