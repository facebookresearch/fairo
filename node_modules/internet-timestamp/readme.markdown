# internet-timestamp

convert javascript date objects into
[rfc3339](http://www.faqs.org/rfcs/rfc3339.html)-compliant timestamp strings

rfc3339 is mandated by
[atom rss](http://www.atomenabled.org/developers/syndication/).

[![browser support](https://ci.testling.com/substack/internet-timestamp.png)](http://ci.testling.com/substack/internet-timestamp)

[![build status](https://secure.travis-ci.org/substack/internet-timestamp.png)](http://travis-ci.org/substack/internet-timestamp)

# example

``` js
var timestamp = require('internet-timestamp');
var d = new Date('Thu Mar 14 19:16:19 2013 -0700');
console.log(timestamp(d));
```
```
$ node example/stamp.js
2013-03-14T19:16:19-07:00
```

# methods

``` js
var timestamp = require('internet-timestamp')
```

## timestamp(date)

Return a
[rfc3339](http://www.faqs.org/rfcs/rfc3339.html)-compliant timestamp from
the Date instance or Date()-parseable string `date`.

# install

With [npm](https://npmjs.org) do:

```
npm install internet-timestamp
```

To use this module in the browser, use [browserify](https://browserify.org).

# license

MIT
