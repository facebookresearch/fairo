import React, { useEffect } from "react";
import { DatePicker, Input, Select } from 'antd';
const { RangePicker } = DatePicker;
const { Option } = Select;

const TimeFilter = (props) => {
    const setFilterType = props.setFilterType;
    const filterOnTime = props.filterOnTime;

    useEffect(() => {}, [props.filterType]);

    return 
};

export default TimeFilter;