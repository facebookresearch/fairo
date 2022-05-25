var http = require('http');
var path = require('path');
var glog = require('../')('/tmp/glog-repo');
var ecstatic = require('ecstatic')(path.join(__dirname, '..', 'static'));

var server = http.createServer(function (req, res) {
    if (glog.test(req.url)) {
        glog(req, res);
    }
    else ecstatic(req, res);
});
server.listen(5001);
