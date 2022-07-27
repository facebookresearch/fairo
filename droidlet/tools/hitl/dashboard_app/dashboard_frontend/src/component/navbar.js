/*
Copyright (c) Facebook, Inc. and its affiliates.

Navigation bar for the dashboard app. 
*/
import React, { useState } from "react";
import 'antd/dist/antd.css';
import '../index.css';
import { Menu, Typography } from 'antd';
import { useLocation, useMatch, useNavigate } from "react-router-dom";
import { SUBPATHS } from "../constants/subpaths";

const menuItems = Object.values(SUBPATHS);
const {Title} = Typography;

const NavBar = (props) => {
    const [, setCurrent] = useState('');
    const matched = useMatch("/:subpath/*") ;
    const subpath = matched ? `/${matched.params.subpath}`: useLocation().pathname;
    let navigate = useNavigate();

    const onClick = (evt) => {
        setCurrent(evt.key);
        navigate(`../${evt.key}`, {replace: true});
    }
    
    return (
    <Menu theme="dark" onClick={onClick} selectedKeys={subpath} mode="horizontal"> 
        <div className="logo">
            <Title level={4} style={{"color": "white"}}>HITL Dashboard</Title>
        </div>
         {  
            // navigation bar tabs
             menuItems.map(item => 
                <Menu.Item key={item.key}> 
                    <div className="navbar-title-content">{item.label}</div>
                </Menu.Item>
            )
         }
    </Menu>);    
}

export default NavBar;