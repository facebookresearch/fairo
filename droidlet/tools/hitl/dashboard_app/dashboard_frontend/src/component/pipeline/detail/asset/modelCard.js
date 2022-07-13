/*
Copyright (c) Facebook, Inc. and its affiliates.

The card showing Model infomation of a run. 
**Note: this is just a placeholder for the model card. TODO: add real content of the model card**

Usage:
<ModelCard />
*/
import { Card } from "antd";
import React from "react";

const { Meta } = Card;

const ModelCard = () => {

    return (
        <div style={{ width: '50%' }}>

            <Card title="Model">
                <Meta />
                model here
            </Card>
        </div>);
}

export default ModelCard;