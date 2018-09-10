var url = window.location.href;
var newurl = "http://ml.dask.org" + url.split("/build/html")[1]

htmlstr = "<div class=\"admonition note\">\
    <p class=\"first admonition-title\">Note</p>\
    <p class=\"last\">The documentation has moved to\
                      <a href=" + newurl + ">ml.dask.org</a></p>\
           </div>";

if (url.indexOf("readthedocs") !== -1) {
    $(document).ready(function () {
        $(".section > h1").prepend(htmlstr);
    })
}
