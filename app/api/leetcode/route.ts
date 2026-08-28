type LeetCodeQuestion = {
  questionFrontendId: string;
  title: string;
  translatedTitle?: string | null;
  difficulty: "Easy" | "Medium" | "Hard";
  titleSlug: string;
};

const query = `
  query questionData($titleSlug: String!) {
    question(titleSlug: $titleSlug) {
      questionFrontendId
      title
      translatedTitle
      difficulty
      titleSlug
    }
  }
`;

function linkToSlug(input: string) {
  try {
    const url = new URL(input);
    if (!/(^|\.)leetcode\.(cn|com)$/i.test(url.hostname)) return null;
    return url.pathname.match(/^\/problems\/([^/?#]+)/i)?.[1] || null;
  } catch {
    return null;
  }
}

async function getQuestion(link: string) {
  const slug = linkToSlug(link);
  if (!slug) return { link, error: "不是有效的 LeetCode 题目链接" };

  try {
    const response = await fetch("https://leetcode.cn/graphql/", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "referer": `https://leetcode.cn/problems/${slug}/`,
        "user-agent": "Ling-Learning-Lab/1.0",
      },
      body: JSON.stringify({ query, variables: { titleSlug: slug } }),
      signal: AbortSignal.timeout(12000),
    });
    if (!response.ok) return { link, slug, error: `LeetCode 返回 ${response.status}` };
    const payload = await response.json() as { data?: { question?: LeetCodeQuestion | null } };
    const question = payload.data?.question;
    if (!question) return { link, slug, error: "没有找到对应题目" };
    return {
      link,
      id: Number(question.questionFrontendId),
      title: question.translatedTitle || question.title,
      englishTitle: question.title,
      difficulty: question.difficulty,
      slug: question.titleSlug,
      href: `https://leetcode.cn/problems/${question.titleSlug}/`,
    };
  } catch {
    return { link, slug, error: "暂时无法连接 LeetCode" };
  }
}

export async function POST(request: Request) {
  let links: unknown;
  try {
    ({ links } = await request.json() as { links?: unknown });
  } catch {
    return Response.json({ error: "请求格式无效" }, { status: 400 });
  }

  if (!Array.isArray(links)) return Response.json({ error: "缺少题目链接" }, { status: 400 });
  const uniqueLinks = Array.from(new Set(links.filter((item): item is string => typeof item === "string"))).slice(0, 30);
  const results = await Promise.all(uniqueLinks.map(getQuestion));
  return Response.json({ results });
}
