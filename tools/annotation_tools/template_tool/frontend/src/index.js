/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react';
import ReactDOM from 'react-dom';
import { BrowserRouter as Router, Switch, Route, Link} from 'react-router-dom'
import './index.css';
import App from './App.js'
import Autocomplete from './autoComplete.js'

ReactDOM.render(
    <React.StrictMode>
      <Router>
        <div style={{border: '1px solid black', marginBottom: 10}}>
          <nav>
            <ul>
              <li>
                <Link to="/">Root</Link>
              </li>
              <li>
                <Link to="/autocomplete">Filters Annotator</Link>
              </li>
            </ul>
          </nav>
        </div>
        <Switch>
          <Route path="/autocomplete">
            <Autocomplete />
          </Route>
          <Route path="/">
            <App />
          </Route>
        </Switch>
      </Router>
    </React.StrictMode>,
    document.getElementById('root'),
);
