import { Table, Typography } from 'antd';
import React from 'react';
import { Link } from 'react-router-dom';

const { Title } = Typography;

const fakeData = [
    {
        name: 'some test run 1',
        batch_id: 11122459,
        description: null
    }, {
        name: 'randome test run 2',
        batch_id: 43050947891,
        description: 'some descpriotion'
    }, {
        name: 'this is another run',
        batch_id: 11940599,
        description: 'some descpriotion'
    }, {
        name: 'verbose',
        batch_id: 495803950,
        description: `Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Eget duis at tellus at. Maecenas sed enim ut sem. Sagittis aliquam malesuada bibendum arcu vitae elementum curabitur. Nisi est sit amet facilisis magna etiam tempor orci eu. Tortor at risus viverra adipiscing at in tellus integer feugiat. Eget nunc lobortis mattis aliquam faucibus. Tellus cras adipiscing enim eu turpis. Quisque non tellus orci ac auctor augue. Purus gravida quis blandit turpis. Pulvinar pellentesque habitant morbi tristique senectus et netus et. Quis lectus nulla at volutpat diam ut venenatis. Neque convallis a cras semper auctor neque vitae tempus. Purus viverra accumsan in nisl nisi scelerisque eu ultrices. Donec ac odio tempor orci. Donec ultrices tincidunt arcu non sodales neque sodales ut. Tortor pretium viverra suspendisse potenti nullam. A iaculis at erat pellentesque adipiscing commodo.

        Etiam sit amet nisl purus. Morbi enim nunc faucibus a pellentesque sit. Ut morbi tincidunt augue interdum velit euismod in. Nibh cras pulvinar mattis nunc sed blandit. Tristique senectus et netus et malesuada fames ac. Metus aliquam eleifend mi in. Tortor condimentum lacinia quis vel eros donec ac odio. Eu tincidunt tortor aliquam nulla facilisi. Vestibulum morbi blandit cursus risus. Dolor sed viverra ipsum nunc aliquet bibendum. Mauris a diam maecenas sed enim ut sem viverra aliquet. Neque sodales ut etiam sit. Eget aliquet nibh praesent tristique magna sit amet purus gravida. Elit ut aliquam purus sit amet luctus venenatis lectus. Mollis nunc sed id semper. Magnis dis parturient montes nascetur. Duis convallis convallis tellus id interdum. Sed felis eget velit aliquet sagittis id.`
    }, {
        name: 'randome 2',
        batch_id: 4234009,
        description: '11'
    }, {
        name: 'randome x',
        batch_id: 4234043851231239,
        description: 'llllfekllfk1'
    }, {
        name: 'xxxxyyyy',
        batch_id: 4395882099,
        description: 'fefvolfalfw'
    }, {
        name: 'more kittens',
        batch_id: 111,
        description: 'cats'
    }
];

// prevent undo sorting
const sortDirections = ['ascend', 'descend', 'ascend']

const jobListCols = [
    {
        title: 'Name',
        dataIndex: 'name',
        sorter: (one, other) => (one.name.localeCompare(other.name)),
        sortDirections: sortDirections,
        width: '10%'
    }, {
        title: 'Batch ID',
        dataIndex: 'batch_id',
        sorter: (one, other) => (one.batch_id < other.batch_id),
        sortDirections: sortDirections,
        width: '10%'
    }, {
        title: 'Description',
        dataIndex: 'description',
        sorter: (one, other) => {
            if (one.description && other.description) {
                return one.description.localeCompare(other.description);
            } else {
                return one.description ? -1 : 0;
            }
        },
        sortDirections: sortDirections,
        ellipsis: true
    }, {
        title: 'Action',
        key: 'action',
        width: '8%',
        render: (_, row) => (
            <Link to={`${row.batch_id}`}>View Detail</Link>
        )
    }
]

const JobList = (props) => {
    const pipelineType = props.pipelineType;
    // to do get jobs by pipelineType

    return (
        <>
            <div style={{ "text-align": "left" }}>
                <Title level={5}>
                    View All {pipelineType} Jobs
                </Title>
            </div>
            <Table
                columns={jobListCols}
                dataSource={fakeData}
                scroll={{ y: '80vh' }}
            />
        </>
    );
}

export default JobList;