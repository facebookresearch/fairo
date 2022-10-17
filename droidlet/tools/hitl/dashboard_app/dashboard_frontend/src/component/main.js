import React, {useState, useContext, useEffect, useCallback} from 'react';
import Button from '@mui/material/Button';
/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import {SocketContext} from '../context/socket';
import { List, ListSubheader, ListItem, ListItemText, Divider } from '@mui/material';

const Main = () => {
    const socket = useContext(SocketContext);
    
    const [jobList, setJobList] = useState([]);
    const [loading, setLoading] = useState(false);
    const [jobInfo, setJobInfo] = useState('');
    const [jobTraceback, setJobTraceback] = useState('');

    const handleReceivedJobList = useCallback((data) => {
        setJobList(data);
        setLoading(false);
    }, []);

    const getJobList = () => {
        socket.emit("get_job_list");
        setLoading(true);
    } 

    const handleReciveTraceback = useCallback((data) => {
        setJobTraceback(data);
    }, []);

    const getJobTraceback = () => {
        // hardcoded id for checking backend api connection
        socket.emit("get_traceback_by_id", 20220519194922);
    }

    const handleReciveInfo = useCallback((data) => {
        setJobInfo(JSON.stringify(data));
    }, []);

    const getJobInfo = () => {
        // hardcoded id for checking backend api connection
        socket.emit("get_run_info_by_id", 20211209154235);
    }

    useEffect(() => {
        socket.on("get_job_list", (data) => handleReceivedJobList(data));
        socket.on("get_traceback_by_id", (data) => handleReciveTraceback(data));
        socket.on("get_run_info_by_id", (data) => handleReciveInfo(data));
    }, [socket, handleReceivedJobList, handleReciveTraceback, handleReciveInfo]);

    return (
        <div>
            <Button variant="contained" onClick={() => {console.log(socket)}}>console log</Button>
            <div style={{"display": "flex"}}>
                <div style={{"width" : "400px"}}>
                    <h1>
                    Test list Jobs
                    </h1>
                    <Button variant="contained" onClick={getJobList}>List Jobs</Button>
                    <List       
                        sx={{ width: '100%', maxWidth: 360, border: 'solid gray'}}
                        subheader={
                            <ListSubheader>
                                Job List
                            </ListSubheader>
                    }>  
                        {
                            loading && <div>loading</div>
                        }
                        {
                            jobList.length !== 0 
                            && jobList.map(job => 
                                (<li>
                                    <Divider light />
                                    <ListItem>
                                        <ListItemText>{job}</ListItemText>
                                    </ListItem>
                                </li>
                            ))
                        }
                    </List>
                </div>
                <div  style={{"width" : "400px"}}>
                    <h1>Test get log</h1>
                    <Button variant="contained" onClick={getJobTraceback}>get log</Button>
                    <div>{jobTraceback.length !== 0 && jobTraceback}</div>
                </div>
                <div  style={{"width" : "400px"}}>
                    <h1>test get info</h1>
                    <Button variant="contained" onClick={getJobInfo}>get info</Button>
                    <div>{jobInfo.length !== 0 && jobInfo}</div>
                </div>   
            </div>
        </div>
        );
    
}

export default Main;