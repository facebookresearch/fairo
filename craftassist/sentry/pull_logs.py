import argparse
import os
import requests


def pull_logs(org, project, status, keyword):
    print("Pulling logs...")
    auth_token = os.getenv("SENTRY_AUTH_TOKEN")
    url = "https://sentry.io/api/0/projects/{}/{}/issues/".format(org, project)
    params = {"query": "{} {}".format(status, keyword)}
    headers = {"Authorization": "Bearer {}".format(auth_token)}

    result = []
    response = requests.get(url, params=params, headers=headers)
    result.extend(response.json())

    link = response.headers.get("Link")
    while 'rel="next"; results="true"' in link:
        print("Pulling logs...")
        start = link.find(", <") + 3
        end = link.find('>; rel="next"; results="true";')
        next_link = link[start:end]
        response = requests.get(next_link, headers=headers)
        result.extend(response.json())
        link = response.headers.get("Link")

    return result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--status",
        default="is:unresolved",
        help="status of issues, can be ['is:unresolved', 'is:resolved', 'is:ignored', 'is:assigned', 'is:unassigned']",
    )
    parser.add_argument("--org", default="craftassist", help="sentry organization slug")
    parser.add_argument("--project", default="craftassist", help="sentry project slug")
    parser.add_argument("--keyword", default="", help="search query keyword")
    parser.add_argument("--save_to", default="", help="search result save path")

    args = parser.parse_args()

    result = pull_logs(args.org, args.project, args.status, args.keyword)

    with open(args.save_to, "w") as f:
        for e in result:
            f.write(str(e))
            f.write("\n")
