var git = require('../');
var joinStream = require('join-stream');

var commit = process.argv[2];
var dir = process.argv[3];

git.list(commit, dir)
    .pipe(joinStream('\n'))
    .pipe(process.stdout)
;
