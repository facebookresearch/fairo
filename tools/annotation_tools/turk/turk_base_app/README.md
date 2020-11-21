
# About
This is an example turk-ready react app.

Out of the box a react app doesn't support the turk environment, but with this app you'll be able to compile straight to a single HTML file which can be inserted into the turk interface.

# How to use

## First steps

0) Run the following command:

```
cp -r ../../../dashboard_web/src/components/AnnotationComponents src/prebuilt-components
```

1) Specify the time for your task, and the turk variables in public/index.html script tag. Uncomment any neccesary variables for the task you plan on using.

2) Put your root component in App.js (look for the example tag)

3) import turk from turk.js and call turk.submit from your code when you would like the HIT to end

4) `npm install`

5) `react run-script build`

6) Copy the contents of build/index.html

 ## Uploading to Turk

1) Paste  into the edit project box on turk like so:

```
<HTMLQuestion  xmlns="http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd">
<HTMLContent>
<![CDATA[
*YOUR HTML FILE*
]]></HTMLContent>
<FrameHeight>0</FrameHeight>
</HTMLQuestion>
```
2) Test using an input file and see if your app works before running an large experiments

# More on React and Turk

## Why it doesn’t work out of the box:

Whenever you specify a HTMLQuestion in turk, You’re not actually writing directly to the iframe that renders your HTML, rather you are writing HTML to a deeper node, nested under a form (shown in the picture above). This means that all buttons and other form items that one may want use in the HIT will cause erroneous behavior (such as submitting early).

The next issue is that you will need to submit your data using javascript, instead of using a form. 

The last problem is that you will need to compile your react app to a single page of HTML with all scripts and resources inlined in order to run your task on turk. This process is involved, and requires ejecting your react app and adding some lines to the webpack config.

## Fixing the Form Nesting:

1. Remove the react binding location from index.html (default is a div with id “root”)
2. Add the following to index.js:

```

let r = document.createElement('div')
r.id = "root"
document.getElementsByTagName('body')[0].appendChild(r)

ReactDOM.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
  document.getElementById('root')
);

```

## Submitting Data:

https://github.com/longouyang/mmturkey contains all of the code necessary to submit a HIT with any JS object as the data. Just import and call turk.submit(data)

## Compiling to One File:

https://stackoverflow.com/questions/51949719/is-there-a-way-to-build-a-react-app-in-a-single-html-file worked for me. These changes are permanent so make a backup of your project before hand. 

## Form Elements Required:

Because of the setup that turk uses, you have to have at least one form element in your index.html file (inside react components won’t count). To get around this add a dummy input with the style=“display:hidden;” attribute. You will also need to remove the pre added submit button so that users won’t accidentally submit using the wrong data! You can do this with the following code in your index.js file

```
window.onload = ()=>{
  try {
    document.getElementById('submitButton').remove()
  } catch (error) {
    console.log("Debugging mode")
  }
}
```
