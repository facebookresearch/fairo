import React, {useState, useContext, useEffect, useCallback} from 'react';
import Button from '@mui/material/Button';
import {SocketContext} from '../context/socket';
import { List, ListSubheader, ListItem, ListItemText } from '@mui/material';

const Main = () => {
    const socket = useContext(SocketContext);
    
    const [jobList, setJobList] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleReceived = useCallback((data) => {
        console.log("11111")
        console.log(data)
        setJobList(data)
        setLoading(false)
    }, []);

    const getJobList = () => {
        socket.emit("get_job_list");
        setLoading(true);
    } 

    useEffect(() => {
        socket.on("get_job_list", (data) => handleReceived(data));
    }, [socket, handleReceived]);

    return (
        <div>
            <Button variant="contained" onClick={getJobList}>List Jobs</Button>
            <Button variant="contained" onClick={() => {console.log(socket)}}>console log</Button>
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
                    jobList.length !== 0 && jobList.map(job => (<ListItem><ListItemText>{job}</ListItemText></ListItem>))
                }
            </List>
        </div>
        );
    
}

export default Main;