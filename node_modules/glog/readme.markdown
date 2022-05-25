# glog

git push blog server

# example

## custom http server

Here's what a custom server could look like storing repository data in `./repo`:

``` js
var http = require('http');
var glog = require('glog')(__dirname + '/repo');
var ecstatic = require('ecstatic')(__dirname + '/static');

var server = http.createServer(function (req, res) {
    if (glog.test(req.url)) {
        glog(req, res);
    }
    else ecstatic(req, res);
});
server.listen(5000);
```

## git push glog

First run your http server:

```
$ node server.js
```

Now create a new git repo for articles and set up the remote to point at your
glog server:

```
$ git init
$ git remote add publish http://localhost:5000/blog.git
```

Write an article in markdown,
create an annotated tag for the article,
and push to the git blog server:

```
$ echo -e '# beep\nboop' > robot.markdown
$ git add *.markdown && git commit -m 'initial'
$ glog publish robot.markdown 'this is the title text'
$ git push publish master --tags
```

Now the content should be live on your blog, yay!

## authenticating glog

Continuing from the previous example, we'll add user permissions to our glog
server.

To create a user once you've set the `git remote`, from your blog repo do:

```
$ glog useradd substack
Created user substack
To publish as this user add this remote:

http://substack:42aee89a@localhost:5000/blog.git
```

If you don't already have a remote for the blog repo, pass `--remote=REMOTE` to
the `glog useradd` command.

Once users have been configured, everyone who tries to `git push` new articles
will need to have a user token.

Now you can list the glog users with `glog users`:

```
$ glog users
substack
```

For the rest of the user commands, just type `glog` to see the usage page.

# http api

When you attach a glog handler to your server, these routes are installed:

## /blog.git

Used by [pushover](http://github.com/substack/pushover) to make `git push`
deploys work. You can set this as a git remote and interact with it like any
other git endpoint.

Annotated git tags with the filename as the tag name are used to store title
text, publish date, and which files are "published".

## /blog.json

Return a streaming json array of article metadata for all articles.

Optionally, you can set these query string parameters:

* inline - include the article content bodies along with the document metadata
as `'html'` or `'markdown'`

example output:

```
$ curl localhost:5000/blog.json
[
{"file":"robot.markdown","author":"James Halliday","email":"mail@substack.net","date":"Mon Dec 24 15:31:27 2012 -0800","title":"robots are pretty great","commit":"81c62aa62b6770a2f6bdf6865d393daf05930b4a"}
,
{"file":"test.markdown","author":"James Halliday","email":"mail@substack.net","date":"Mon Dec 24 04:31:53 2012 -0800","title":"testing title","commit":"2a516000d239bbfcf7cdbb4b5acf09486bdf9586"}
]
```

```
 $ curl localhost:5000/blog.json?inline=html
[
{"file":"robot.markdown","author":"James Halliday","email":"mail@substack.net","date":"Mon Dec 24 15:31:27 2012 -0800","title":"robots are pretty great","commit":"81c62aa62b6770a2f6bdf6865d393daf05930b4a","body":"<h1>robots!</h1>\n\n<p>Pretty great basically.</p>"}
,
{"file":"test.markdown","author":"James Halliday","email":"mail@substack.net","date":"Mon Dec 24 04:31:53 2012 -0800","title":"testing title","commit":"2a516000d239bbfcf7cdbb4b5acf09486bdf9586","body":"<h1>title text</h1>\n\n<p>beep boop.</p>\n\n<p><em>rawr</em></p>"}
]
```

## /blog.xml

Return an [atom rss](http://www.atomenabled.org/developers/syndication/) stream
with inline content.

## /blog/$FILE.markdown

Fetch a source document $FILE as markdown.

## /blog/$FILE.html

Fetch a source document $FILE.markdown rendered as html.

# methods

```  js
var glog = require('glog')
```

## var blog = glog(opts)

Create a new `blog` handle using `opts.repodir` to store git blog data.

If `opts` is a string, it's taken as the `opts.repodir`.

You can also set `opts.title` and `opts.id` which are used as defaults by the
rss feed, and `opts.highlight` which is the highlight-function used by marked.

All other `opts` are passed through directly to `marked.parse(src, opts)`.

## blog(req, res)

Handle the `(req, res)` in order to serve blog.json and blog.git.

## blog.get(name)

Get a single article, returning a readable stream of a single blog documents
object. Blog documents have:

* doc.title - title text
* doc.commit - document git commit hash
* doc.date - parseable date string
* doc.author - author name as a string
* doc.email - author email from git commit data
* doc.file - document filename in the git repo

## blog.list(opts)

Return a readable stream of blog article documents.

Optionally:

* `opts.limit` - number of results to show
* `opts.start` - show results starting at this tag or title
* `opts.after` - show results after this tag or title

## blog.read(file)

Return a readable stream with the contents of `file`.

## blog.inline(format)

Return a through stream you can pipe `blog.list()` to that will inline article
contents rendered in `format`: either `'html'` or `'markdown'`.

`.inline()` adds a `doc.body` string with the article contents to the document
object.

## blog.test(req.url)

Return whether or not to defer to `blog` for handling routes.

## blog.rss(opts)

Return an [atom rss](http://www.atomenabled.org/developers/syndication/) stream
with the blog content inlined in `<content>` tags.

opts are the required elements from the atom spec but you can probably ignore
them and it will still work:

* opts.id - just use your blog address or domain name
* opts.title - blog title to use in the feed

# usage

```
usage:

  glog publish FILE "TITLE..."

    Publish FILE with TITLE by creating an annotated tag.

  glog users

    Show the list of glog users.

  glog useradd USER

    Generate an auth token for USER to use as a git remote.

  glog userdel USER

    Delete a USER.

  glog token USER

    Show the git remote token for USER.

```

# install

With [npm](https://npmjs.org), to get the `glog` command do:

```
npm install -g glog
```

and to get the library do:

```
npm install glog
```

# license

MIT
