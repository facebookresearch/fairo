/*
Copyright (c) Facebook, Inc. and its affiliates.
*/
import React from 'react'
import * as d3 from 'd3'
import SegmentRenderer from './AnnotationComps/SegmentRenderer'
import {getInfo, clean, area} from './CocoHelpers.js'

/*
Defines a react component that provides the user an interface to upload a csv from turk and to 
visualize the results of a react experiment, and to export the results to a pycoco's dataset
*/

const COLORS = [
    "rgba(0,200,0,.5)",
    "rgba(200,0,0,.5)",
    "rgba(0,100,255,.5)",
    "rgba(255,150,0,.5)",
    "rgba(100,255,200,.5)",
    "rgba(200,200,0,.5)",
    "rgba(0,200,150,.5)",
    "rgba(200,0,200,.5)",
    "rgba(0,204,255,.5)",
];

class CSVInput extends React.Component {
    constructor(props){
        super(props)

        this.state = {
            data: [],
            pos: 0,
            isExporting: false,
            dataLoaded: false
        }

        this.renderedSegment = ""

        this.generateCocoDataset = this.generateCocoDataset.bind(this)
    }

    componentDidMount(){
        let fileInput = document.getElementById("csv-input")
        fileInput.onchange = (e)=>{
            var f = new FileReader()
            f.onload = ()=>{
                let data = undefined;
                try {
                    let data = d3.csvParse(f.result)
                } catch (error) {
                    alert("Invalid CSV File")
                }
                if (data != undefined) {
                    this.setState({
                        data: data.map((x)=>{
                            if(x["Answer.data"]){
                                let i = JSON.parse(x["Answer.data"])
                                i['url'] = x["Input.image_url"]
                                i['workerId'] = x['WorkerId']
                                i['time'] = x['WorkTimeInSeconds']
                                i['status'] = x['AssignmentStatus'] //Approved/Rejected/Submitted
                                return i
                            } else {
                                return null
                            }
                        }),
                        dataLoaded: true
                    }, ()=>{
                        this.renderSegment()
                    })
                }
                
            }
            f.readAsText(fileInput.files[0])
        }
    }

    renderSegment(){
        if (!this.state.data[this.state.pos]) {
            return
        }
        this.currentImg = new Image()
        this.currentImg.onload = ()=>{
            this.renderedSegment = 
            <div>
                <p>{this.state.data[this.state.pos].workerId}: {this.state.data[this.state.pos].time/60}</p>
                <SegmentRenderer
                    objects = {this.state.data[this.state.pos].objectIds}
                    pointMap = {this.state.data[this.state.pos].points}
                    nameMap = {this.state.data[this.state.pos].names}
                    propertyMap = {this.state.data[this.state.pos].properties}
                    colors = {COLORS}
                    img = {this.currentImg}
                />
            </div>
            this.forceUpdate()
        }
        this.currentImg.src = this.state.data[this.state.pos].url
    }

    render(){
        return(
        <div>
            <input type="file" id="csv-input"/>
            <button onClick={()=>{
                this.setState({
                    pos: (this.state.pos + 1) % this.state.data.length
                }, this.renderSegment)
            }}>Next</button>
            <button
                onClick={this.exportDataset.bind(this)}>Export to Dataset</button>
            {this.renderedSegment}
        </div>
        )
    }

    exportDataset(){
        if(!this.state.dataLoaded){
            alert("Data is not loaded, please wait")
            return
        }

        var data = this.generateCocoDataset()
        
        // saving code taken from here: https://jsfiddle.net/koldev/cW7W5/
        var saveData = (function () {
            var a = document.createElement("a");
            document.body.appendChild(a);
            a.style = "display: none";
            return function (data, fileName) {
                var json = JSON.stringify(data),
                    blob = new Blob([json], {type: "octet/stream"}),
                    url = window.URL.createObjectURL(blob);
                a.href = url;
                a.download = fileName;
                a.click();
                window.URL.revokeObjectURL(url);
            };
        }());

        var fileName = "dataset.json";
        saveData(data, fileName);
        // End external code
    }

    generateCocoDataset(){
        let imageIdCounter = 200000 //Max of 300000 images seems reasonable
        let categoryIdCounter = 0
        let annotationIdCounter = 500000
        var images = {}
        var annotations = []
        var categories = {}

        this.state.data.map((x)=>{
            // Process Image
            let license = 0
            let filename = x['url'].split('/').pop()
            let coco_url = x['url']
            let height = x.metaData.height
            let width = x.metaData.width

            if(!images[filename]){
                var imageId = imageIdCounter
                imageIdCounter += 1

                images[filename] = {
                    license: license,
                    file_name: filename,
                    coco_url: coco_url,
                    height: height,
                    width: width,
                    data_captured: null,
                    flicker_url: coco_url,
                    id: imageId
                }
            } else {
                imageId = images[filename].id
            }
            // Process Annotations
            for(const objectId in x.objectIds){
                let name = clean(x.names[objectId])

                if(!categories[name]){
                    var catId = categoryIdCounter
                    categoryIdCounter += 1

                    categories[name] = {
                        supercategory: name,
                        id: catId,
                        name: name
                    }
                } else {
                    catId = categories[name].id
                }

                let points_raw = x.points[objectId]
                let smallestX = +Infinity
                let smallestY = +Infinity
                let biggestX = -Infinity
                let biggestY = -Infinity
                let segmentation = [points_raw.flatMap((pt)=>{
                    smallestX = Math.min(smallestX, pt.x)
                    smallestY = Math.min(smallestY, pt.y)
                    biggestX = Math.max(biggestX, pt.x)
                    biggestY = Math.max(biggestY, pt.y)
                    return [pt.x, pt.y]
                })]
                let bbox = [smallestX, smallestY, biggestX-smallestX, biggestY-smallestY]
                let annID = annotationIdCounter
                annotationIdCounter += 1
                annotations.push({
                    id: annID,
                    image_id: imageId,
                    category_id: catId,
                    segmentation: segmentation,
                    area: area(points_raw),
                    bbox: bbox,
                    iscrowd: 0
                })
            }
        })

        let imageArray = []
        for(const filename in images){
            imageArray.push(images[filename])
        }

        let categoryArray = []
        for(const name in categories){
            categoryArray.push(categories[name])
        }

        return {
            info: getInfo(),
            images: imageArray,
            annotations: annotations,
            licenses: [{id: 0,
                name: "MIT",
                url: "https://opensource.org/licenses/MIT"}],
            categories: categoryArray
        }
    }
}

export default CSVInput