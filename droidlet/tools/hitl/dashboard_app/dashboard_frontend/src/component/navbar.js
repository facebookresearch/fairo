import React, { useState } from "react";
import 'antd/dist/antd.css';
import { Menu } from 'antd';
import { useLocation, useNavigate } from "react-router-dom";
import { SUBPATHS } from "../constants/subpaths";

const menuItems = Object.values(SUBPATHS);

const NavBar = (props) => {
    const [, setCurrent] = useState('');
    const location = useLocation();
    let navigate = useNavigate();

    const onClick = (evt) => {
        setCurrent(evt.key);
        navigate(`../${evt.key}`, {replace: true});
    }

    console.log(location)

    return <Menu onClick={onClick} selectedKeys={location.pathname} mode="horizontal" items={menuItems} />;    
}

export default NavBar;