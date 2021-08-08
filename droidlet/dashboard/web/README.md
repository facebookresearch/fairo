# Dashboard app

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Know-hows of the app

**Note: Run all commands from the `droidlet/dashboard/web` folder.**

### Installing dependencies

```
npm install -g yarn
yarn install
```

### Starting the app for local development

```
yarn start
```

This will run the app in development mode.<br />
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.
The page will reload if you make edits.You will also see any lint errors in the console. <br />
For debugging: you can set up the debugger in your IDE (for example [here](https://code.visualstudio.com/docs/nodejs/reactjs-tutorial#_debugging-react)) or use the developer console on your browser (for example: Command + option + J for Chrome on Macbook)

### Accessing Mobile Dashboard

After running yarn start, open [http://localhost:3000/mobile.html](http://localhost:3000/mobile.html) to view it in the browser. Use the Chrome Developer tools to 
view it in mobile <br />

To access on a physical device, find your computers ip address. Open http://[ip address]:3000/mobile.html on your physical phone <br />

If you are running into issues, look at this link: [https://docs.google.com/document/d/1chTdtbW2t1swpWe4NMfyxn31tK_Bta3ZXwLfgM9w5rs/edit?usp=sharing](https://docs.google.com/document/d/1chTdtbW2t1swpWe4NMfyxn31tK_Bta3ZXwLfgM9w5rs/edit?usp=sharing)

### Mobile Dashboard Codeflow

When going to http://localhost:3000/mobile.html, react will serve the component in mobile.js (the mobile entry point), that renders MobileMainPane.js

MobileMainPane.js controls what is being displayed, all mobile bugs can be found following the code in MobileMainPane and the components that are used in it


### Files to check-in

Check-in the following files or folders:

```
droidlet/dashboard/web/src/
droidlet/dashboard/web/yarn.lock
droidlet/dashboard/web/package.json
droidlet/dashboard/web/build/
droidlet/dashboard/web/public/
```

Do not check-in the following at any stage:

```
droidlet/dashboard/web/node_modules
droidlet/dashboard/web/package-lock.json
node_modules
yarn.lock # in any other folder than droidlet/dashboard/web/yarn.lock
```

#### Checking in your changes

Once you've made changes to the project:
1. Create a Pull Request with all the code changes for review.
2. Then run `yarn build` to update the `build` folder. Run: `git add -f build/` to check in your `build` files and submit this as a separate PR.

### Adding or removing package dependencies

```
yarn add [package name]
yarn remove [package name]
```

### Creating production builds

```
yarn build
```

This builds the app for production to the `build` folder.<br />
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.<br />
Your app is now ready to be deployed!


### Unit tests and individual component library via Storybook

To start the storybook to see the individual react components, run :
```
yarn storybook
```


To run tests, run :

```
yarn test
```

This launches the test runner in the interactive watch mode.<br />
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

## Errors FAQ

### Error on `yarn test`

If you get an error on `yarn test` that says `jest-haste-map: Watchman crawl failed. Retrying once with node crawler.`, do `sudo mv /usr/local/bin/watchman /usr/local/bin/watchman.bak`


### `yarn build` fails to minify

This section has moved here: https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify


## Directory structure

1. All the source code files are in the `src` folder
2. Build files in the `build` folder.
3. `package.json` and `yarn.lock` are the dependency files. `package.json` has the name and version of the dependencies and `yarn.lock` is the lock file. When adding a new package using `yarn add` these are the files that get updated.
4. `public` folder contains the html file for the project.

## Contributing to the app

### New React component
If you want to add a new React component to the project:
1. Add the corresponding React component `js` and `css` files to the [components folder under `src`](https://github.com/facebookresearch/droidlet/tree/main/droidlet/dashboard/web/src/components).
2. Add your component to [index.js](https://github.com/facebookresearch/droidlet/blob/main/droidlet/dashboard/web/src/index.js) to render it in the layout. If you want to add another stacked column check how `Memory 2D` component is added in the layout to be rendered on top right (or `Navigator` on bottom right). The ordering of components in `index.js` is consistent with how these are rendered in the frontend.
3. Manage and set the state of your component [in `StateManager`](https://github.com/facebookresearch/droidlet/blob/main/droidlet/dashboard/web/src/StateManager.js). 
  - An example snippet of how to receive and process information from backend on this component :
    1. First define the method in `StateManager` :
       ```
       setComponent(result) {
           // Some code here that sets the state of the
           // component or does something with result
       } 
    2. Now register this method with `StateManager` class in the constructor :
        ```
        this.setComponent = this.setComponent.bind(this);
        ```
    3. Now register this method with socker event in `StateManager`'s constructor:
        ```
        socket.on("setComponentState", this.setComponent);
        ```
      Your method will now be called when the backend sends the socket event : `"setComponentState"`.
  - An example of how to send information from this component to backend is [here](https://github.com/facebookresearch/droidlet/blob/main/droidlet/dashboard/web/src/components/QuerySemanticParser.js#L29). The individual components should always send socket events via the `StateManager`.  
4. Add new dependencies introduced by your component to [package.json](https://github.com/facebookresearch/droidlet/blob/main/droidlet/dashboard/web/package.json) and [yarn.lock](https://github.com/facebookresearch/droidlet/blob/main/droidlet/dashboard/web/yarn.lock) using `yarn add [package name]`.

### Frontend changes

Most frontend changes when made should be reflected in the app. The app at [http://localhost:3000](http://localhost:3000) will reload if you make frontend edits.<br />
You will also see any errors in the console.

Note that the dashboard serving at : [127.0.0.1:8000](127.0.0.1:8000) by default serves the `build` folder on frontend and will not reflect these local changes until the `build` folder is updated.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: https://facebook.github.io/create-react-app/docs/code-splitting

### Analyzing the Bundle Size

This section has moved here: https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size

### Making a Progressive Web App

This section has moved here: https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app

### Advanced Configuration

This section has moved here: https://facebook.github.io/create-react-app/docs/advanced-configuration

### Deployment

This section has moved here: https://facebook.github.io/create-react-app/docs/deployment