import './App.css';
import React from 'react';
import {SocketContext, socket} from './context/socket';
import Main from './component/main';

function App() {
  return (
    <SocketContext.Provider value={socket}>
      <div className="App">
        <h1>
          Dashboard
        </h1>
        <Main />
      </div>
    </SocketContext.Provider>
  
  );
}

export default App;
