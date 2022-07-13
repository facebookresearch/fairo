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