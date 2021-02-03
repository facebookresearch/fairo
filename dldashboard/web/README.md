# Dashboard app

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

## Know-hows of the app

- To install the app first, do: 
```
npm install -g yarn ; yarn upgrade
```

- Now to start the app, run: `yarn start`.

This will run the app in development mode.<br />
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.<br />
You will also see any lint errors in the console.

- To start the storybook to see the individual react components, run : `yarn storybook`

- To run tests, run : `yarn test`. <br />

This launches the test runner in the interactive watch mode.<br />
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

- To add a new package to the project, run : `yarn add [package name]`
- To remove a package from the project: `yarn remove [package name]`
- To generate build files, run : `yarn build`

This builds the app for production to the `build` folder.<br />
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.<br />
Your app is now ready to be deployed!


## Errors 

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
1. Add the React component `js` and `css` files to the [components folder under src](https://github.com/facebookresearch/droidlet/tree/main/dldashboard/web/src/components).
2. Add the component to [index.js](https://github.com/facebookresearch/droidlet/blob/main/dldashboard/web/src/index.js) to render it in the layout.
3. Manage and set the state of your component [here](https://github.com/facebookresearch/droidlet/blob/main/dldashboard/web/src/StateManager.js)
4. Add new dependencies introduced by your component to [package.json](https://github.com/facebookresearch/droidlet/blob/main/dldashboard/web/package.json) and [yarn.lock](https://github.com/facebookresearch/droidlet/blob/main/dldashboard/web/yarn.lock) using `yarn add [package name]`.

### Frontend changes

Most frontend changes when made should be reflected in the app. The app at [http://localhost:3000](http://localhost:3000) will reload if you make frontend edits.<br />
You will also see any errors in the console.

Note that the dashboard serving at : [127.0.0.1:8000](127.0.0.1:8000) by default serves the `build` folder on frontend and will not reflect these local changes until the `build` folder is updated.

### Checking in your changes

Once you've made changes to the project:
1. Create a Pull Request with all the code changes for review.
2. Then run `yarn build` to update the `build` folder. Run: `git add -f build/` to check in your `build` files and submit this as a separate PR.

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
