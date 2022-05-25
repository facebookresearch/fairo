var test = require('tap').test;
var git = require('../');
var through = require('through');

test('newer directory list', function (t) {
    t.plan(1);
    
    var commit = '69c368dddd0aece575f7c874a61239d8194ca7f7';
    var dir = 'example';
    
    var data = [];
    var tr = through(
        function (file) { data.push(file) },
        function () {
            t.same(data.sort(), [ 'read.js', 'list.js' ].sort());
        }
    );
    
    git.list(commit, dir).pipe(tr);
});

test('older directory list', function (t) {
    t.plan(1);
    
    var commit = '1b37f60da787df42edb863e5ea726e4dd7ce0457';
    var dir = 'example';
    
    var data = [];
    var tr = through(
        function (file) { data.push(file) },
        function () {
            t.same(data, [ 'x.js' ]);
        }
    );
    
    git.list(commit, dir).pipe(tr);
});
