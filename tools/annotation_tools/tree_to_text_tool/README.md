## Tree to text tool

This tool shows the user a depth 1 n-ary tree and asks the user to describe the intent of the tree using pure text.

This could be any representation : tree / graph / any other logical form. The idea is to have people describe the meaning
of the representation using text.

Example:
For:                        
```
Build (intended action)
│   
│
└───big wooden house (the thing to be built)
│   
│   
└───behind me (the place where the build will happen)
```           

desired inputs would be :
- `please construct a big house made out of wood at the back of where I am standing` or
- `can you make me a house using wood and put it behind me. Make it big please.`
- `build a big wooden house behind me.`
- `using wood, create a big house and make it behind me.`

and we can use the same tool for subcomponents as well, so:
```
house (name)
│   
│
└───big (the size)
│   
│   
└───wooden (the material)
```
desired inputs for this tree would be:
- `a house made out of wood and that is big`
- `a big wooden home`
- `big house constructed out wood` etc

and these subcomponents are composable.

## The tool
As of now, you can just open the file: `tree_to_text_wip.html` in a browser.

- An upcoming functionality is : having users click on parts of sentence that correspond to different
  nodes in the tree.
