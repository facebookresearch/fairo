#!/usr/bin/env node
var fs = require('fs');
var path = require('path');
var spawn = require('child_process').spawn;
var exec = require('child_process').exec;
var run = require('comandante');
var url = require('url');
var EventEmitter = require('events').EventEmitter;
var hyperquest = require('hyperquest');
var concat = require('concat-stream');

var argv = require('optimist').argv;
var cmd = argv._[0];
var mkdirp = require('mkdirp');

if (cmd === 'publish') {
    var file = argv._[1];
    var title = argv._.slice(2).join(' ');
    console.log('# git tag ' + file + ' -m ' + JSON.stringify(title));
    spawn('git', [ 'tag', file, '-m', title ], { stdio : [ 0, 1, 2 ] });
}
else if (cmd === 'useradd') {
    var user = argv._[1] || process.env.USER;
    getRemote(function (err, remote) {
        if (err) return console.error(err);
        var uri = remote.replace(/\.git$/, '')
            + '/_/useradd/' + user
        ;
        hyperquest(uri).pipe(concat(function (buf) {
            var token = String(buf);
            if (!/^\w+\s*$/.test(token)) return console.log(token);

            var u = url.parse(remote);
            console.log(u.protocol + '//'
                + encodeURIComponent(user)
                + ':' + encodeURIComponent(token.trim())
                + '@' + u.host + u.pathname
            );
        }));
    });
}
else if (cmd === 'userdel') {
    var user = argv._[1] || process.env.USER;
    getRemote(function (err, remote) {
        if (err) return console.error(err);
        var uri = remote.replace(/\.git$/, '')
            + 'blog/_/userdel/' + user
        ;
        hyperquest(uri).pipe(process.stdout);
    });
}
else if (cmd === 'users') {
    getRemote(function (err, remote) {
        if (err) return console.error(err);
        var uri = remote.replace(/\.git$/, '') + '/_/users';
        hyperquest(uri).pipe(process.stdout);
    });
}
else {
    fs.createReadStream(path.join(__dirname, '/usage.txt'))
        .pipe(process.stdout)
    ;
}

function error (msg) {
    console.error(msg);
    process.exit(1);
}

function getRemote (cb) {
    if (argv.remote) return cb(null, argv.remote);

    exec('git remote -v', function (err, stdout, stderr) {
        if (err) return cb(err);
        var m = /(https?:\/\/\S+\/blog\.git)\b/.exec(stdout);
        if (!m) return cb('no blog.git http remote');
        cb(null, m[1]);
    });
}
