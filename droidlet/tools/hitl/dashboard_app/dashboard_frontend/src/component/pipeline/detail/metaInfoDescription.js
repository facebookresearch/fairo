/*
Copyright (c) Facebook, Inc. and its affiliates.

Desciption component that shows details of meta infomation of a specific run. 
*/
import { Descriptions } from "antd";
import DescriptionsItem from "antd/lib/descriptions/Item";
import React from "react";
import { METADATA_CONSTANTS, METADATA_ORDER } from "../../../constants/runContants";

const MetaInfoDescription = (props) => {
    let metaInfo = Object.entries(props.metaInfo).filter((o) => o[0] in METADATA_CONSTANTS);
    metaInfo.sort((one, other) => METADATA_ORDER.indexOf(one[0]) - METADATA_ORDER.indexOf(other[0]));

    const getDesciptionText = (o) => {
        if (!o[1]) {
            return "NA";
        } else if (o[0].includes("_TIME")) {
            const idx = o[1].indexOf(".");
            return o[1].substring(0, idx);
        } 
        return o[1];
    }

    return <div>
        <Descriptions>
            {metaInfo.map((o) =>
                <DescriptionsItem
                    label={METADATA_CONSTANTS[o[0]].label}
                    span={METADATA_CONSTANTS[o[0]].span}>
                    {getDesciptionText(o)}
                </DescriptionsItem>
            )}
        </Descriptions>
    </div>;
}

export default MetaInfoDescription;