var http = require('http');
var JSONStream = require('JSONStream');
var through = require('through');

http.get({ path : '/blog.json?inline=html' }, function (res) {
    var parser = JSONStream.parse([ true ]);
    parser.pipe(through(function (doc) {
        var div = createArticle(doc);
        document.body.appendChild(div);
    }));
    res.pipe(parser);
});

function createArticle (doc) {
    var div = document.createElement('div');
    
    var title = document.createElement('div');
    var anchor = document.createElement('a');
    var name = doc.title.replace(/[^A-Za-z0-9]+/g,'_');
    anchor.setAttribute('name', name);
    anchor.setAttribute('href', '#' + name);
    anchor.textContent = doc.title;
    title.appendChild(anchor);
    div.appendChild(title);
    
    var date = document.createElement('div');
    date.textContent = doc.date;
    div.appendChild(date);
    
    var body = document.createElement('div');
    div.appendChild(body);
    
    body.innerHTML = doc.body;
    
    return div;
}
