## Test SPIM data CEBRA model

- Use CEBRA time contrastive learning on neural data from one fish
    - design model
    - convert SPIM data to usable format 
    - load data
    - fit
    - plot embeddings
    - try to decode stimulus frame (for spots only)
        - Doing this naively should fail because there won't be any change in activity on the exact frame<br/><br/>
    - try to decode stimulus type (left/right spots)
        - create a discrete variable that labels the peri-stimulus frames for right and left spots
        - This should inform the decoder to separate embedding states (which should vary between left and right spots)<br/><br/>
    - try to decode eye position and compare it to the ElasticNet models
        - decoding of eye position (or tail vig) should work a lot better than stimulus response, because our discrete variables are very sparse<br/><br/>
    - try to decode response?<br/><br/>

- Use CEBRA variable contrastive learning on neural data from one fish, for decoding stimulus type
    - provide stimulus presentation type (left or right)
    - See if this can be used to decode stimulus type
    - run a parameter sweep using grid search
    - "hypothesis test" with also providing whether fish responded and whether fish were swimming during presentation
    - compute InfoNCE loss to see which may provide best embeddings
    - again, see if these can be used to decode stimulus type
