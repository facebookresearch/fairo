import React from "react";
import Grid from "@material-ui/core/Grid";
import { Typography } from "@material-ui/core";

function DetailRow(props) {
  if (null === props.value || undefined === props.value) {
    return null;
  }
  return (
    <>
      <Grid item xs={4} s={4}>
        <Typography variant="button" align="right">
          {props.title}
        </Typography>
      </Grid>
      <Grid item xs={8} s={8}>
        <Typography variant="body1" align="left">
          {props.value}
        </Typography>
      </Grid>
    </>
  );
}

class MemoryDetail extends React.Component {
  render() {
    const [, , getMemoryForUUID] = this.props.memoryManager;

    const memory = getMemoryForUUID(this.props.uuid);

    return (
      <Grid container spacing={2} style={{ padding: "16px" }}>
        <Grid item xs={11}>
          <Typography variant="h4" align="left">
            Attributes
          </Typography>
        </Grid>
        <DetailRow title="UUID" value={memory.data[0]} />
        <DetailRow title="ID" value={memory.data[1]} />
        <DetailRow title="X" value={memory.data[2]} />
        <DetailRow title="Y" value={memory.data[3]} />
        <DetailRow title="Z" value={memory.data[4]} />
        <DetailRow title="Yaw" value={memory.data[5]} />
        <DetailRow title="Pitch" value={memory.data[6]} />
        <DetailRow title="Type" value={memory.data[9]} />
        <DetailRow title="Name" value={memory.data[7]} />
        <Grid item xs={12}>
          <Typography variant="h4" align="left">
            Triples
          </Typography>
        </Grid>

        <Grid item xs={4}>
          <Typography variant="button" align="left">
            Predicate
          </Typography>
        </Grid>
        <Grid item xs={4}>
          <Typography variant="button" align="left">
            Value
          </Typography>
        </Grid>
        <Grid item xs={4}>
          <Typography variant="button" align="left">
            Confidence
          </Typography>
        </Grid>

        {memory.triples.map((triple) => {
          return (
            <>
              <Grid item xs={4}>
                <Typography variant="body2" align="left">
                  {triple[4]}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="left">
                  {triple[6]}
                </Typography>
              </Grid>
              <Grid item xs={4}>
                <Typography variant="body2" align="left">
                  {triple[7]}
                </Typography>
              </Grid>
            </>
          );
        })}
      </Grid>
    );
  }
}

export default MemoryDetail;
