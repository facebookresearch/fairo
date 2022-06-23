import React, {useState, useContext, useEffect, useCallback} from 'react';
import Button from '@mui/material/Button';
import {SocketContext} from '../context/socket';

const Main = () => {
    const socket = useContext(SocketContext);
    console.log(socket);
    socket.on("connect", () => {
        console.log(socket.connected); // true
      });

    const getJobList = () => {
        socket.emit("my message", "hello");
        // setJobList(['j1', 'j2', 'j3'])
      }
    
    const [jobList, setJobList] = useState([]);

    const handleReceived = useCallback((data) => {
        console.log(data)
        setJobList(data)
    }, []);

    useEffect(() => {
        socket.on("my message", (data) => handleReceived(data));
    }, [socket, handleReceived]);

    return (
        <div>
            <Button variant="contained" onClick={getJobList}>List Jobs</Button>
            <Button variant="contained" onClick={() => {console.log(socket)}}>console log</Button>
            <div>
            Job List
            <div>
                {
                jobList.length !== 0 && jobList.map(job => (<div>{job}</div>))
                }
            </div>
            </div>
        </div>
        );
    
}

export default Main;