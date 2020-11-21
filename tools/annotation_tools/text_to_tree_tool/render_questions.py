"""
Copyright (c) Facebook, Inc. and its affiliates.
"""

import re


def render_q(q, parent_id, show=True, show_siblings=True, sentence_id=""):
    """Return a fieldset for the given question"""
    assert "key" in q, "Missing key for q: {}".format(q)
    q_id = "{}.{}".format(parent_id, q["key"])
    r = ""
    r += '<fieldset id="{}" style="display:{}">'.format(q_id, "block" if show else "none")
    r += label_tag(tooltip=q.get("tooltip")) + q["text"] + "</label>"
    if "radio" in q:
        r += render_radios(
            q["radio"],
            q_id,
            add_other_opt=q.get("add_radio_other", True),
            show_siblings=show_siblings,
            sentence_id=sentence_id,
        )
    if "span" in q:
        r += render_span(q_id, q.get("optional"), sentence_id=sentence_id)
    r += "</fieldset>"
    return r


def render_span(parent_id, optional=False, sentence_id=""):
    r = ""
    group_id = "{}.span".format(parent_id)
    if optional:
        onclick = """var x = document.getElementById('{}');
          x.style.display = x.style.display == 'block' ? 'none' : 'block';""".format(
            group_id
        )
        r += """<label class="btn btn-primary btn-sm" onclick="{}"
          style="margin-left:10px">Click and select all words if specified</label>""".format(
            onclick
        )
    r += '<div id="{}" class="btn-group" data-toggle="buttons" style="display:{}">'.format(
        group_id, "none" if optional else "block"
    )
    for i in range(40):
        input_id = "{}#{}".format(group_id, i)
        r += """<label class="btn btn-default word{j}{i}"
             name="{input_id}">""".format(
            input_id=input_id, i=i, j=sentence_id
        )
        r += '<input type="checkbox" autocomplete="off" id="{input_id}" \
            name="{input_id}">${{word{j}{i}}}'.format(
            input_id=input_id, i=i, j=sentence_id
        )
        r += "</label>"
    r += "</div>"
    return r


def render_radios(opts, parent_id, add_other_opt=True, show_siblings=True, sentence_id=""):
    if add_other_opt:
        opts = opts + [{"text": "Other", "key": "Other"}]

    r = ""
    suffix = ""

    for opt in opts:

        opt_id = "{}.{}".format(parent_id, opt["key"])
        nexts = opt.get("next", [])

        # render child questions
        suffix += (
            '<div id="{}.next" style="display:none">'.format(opt_id)
            + "\n".join([render_q(n, opt_id, sentence_id=sentence_id) for n in nexts])
            + "</div>"
        )

        # get onchange function
        sibling_ids = ["{}.{}".format(parent_id, o["key"]) for o in opts]
        # child_ids = ["{}.{}".format(opt_id, n["key"]) for n in nexts]
        onchange = "\n".join(
            [
                """
                console.log('Toggling {sid}');
                if (document.getElementById('{sid}.next')) {{
                  document.getElementById('{sid}.next').style.display = \
                    document.getElementById('{sid}').checked ? 'block' : 'none';
                }}

            """.format(
                    sid=sid
                )
                for sid in sibling_ids
            ]
        )
        if not show_siblings:
            onchange += "\n".join(
                [
                    """
                    console.log('Hiding siblings {sid}');
                    if (document.getElementById('div_{sid}')) {{
                      document.getElementById('div_{sid}').style.display = \
                      document.getElementById('{sid}').checked ? 'block' : 'none';
                    }}
                """.format(
                        sid=sid
                    )
                    for sid in sibling_ids
                ]
            )

        # produce div for single option
        r += '<div class="radio" id="div_{}">'.format(opt_id) + label_tag(opt.get("tooltip"))
        r += """<input name="{}"
                       id="{}"
                       type="radio"
                       value="{}"
                       onchange="{}"
              />""".format(
            parent_id, opt_id, opt["key"], onchange
        )
        r += opt["text"]
        r += "</label></div>"

    return r + suffix


def label_tag(tooltip=None):
    if tooltip:
        return '<label data-toggle="tooltip" data-placement="right" title="{}">'.format(tooltip)
    else:
        return "<label>"


def child_id(parent_id, text):
    return parent_id + "." + re.sub(r"[^a-z]+", "-", text.lower().strip()).strip("-")
