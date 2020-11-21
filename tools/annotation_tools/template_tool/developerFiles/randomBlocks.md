# Random blocks

This block consists of 2 fields: `randomCategories` and `name`. 
Currently, the user enters `", " delimted names of template objects to random over.

# Changing the delimiter

To change the delimiter in the template tool interface, 
`cd frontend/src/helperFunctions/generateCodeAndSurface.js`.

and then change 
```
    const randomChoices = element
          .getFieldValue('randomCategories')
          .split(', ');
```


Also, 
`cd frontend/src/saveToLocalStorage/saveRandom.js`
and change
```  
    const randomOver = block.getFieldValue
    ('randomCategories').split(', ');
```
