import { getPageMap } from "nextra/page-map";
import { Footer, Layout } from "nextra-theme-docs";

import "nextra-theme-docs/style.css";
import { Header } from "@/components/landing/header";
import { getLocaleByLang } from "@/core/i18n/locale";

export const metadata = {
  // Define your metadata here
  // For more information on metadata API, see: https://nextjs.org/docs/app/building-your-application/optimizing/metadata
};

const footer = <Footer>MIT {new Date().getFullYear()} © Nextra.</Footer>;

const i18n = [
  { locale: "en", name: "English" },
  { locale: "zh", name: "中文" },
];

export default async function DocLayout({ children, params }) {
  const { lang } = await params;
  const locale = getLocaleByLang(lang);
  return (
    <Layout
      navbar={
        <Header
          className="relative max-w-full px-10"
          homeURL="/"
          locale={locale}
        />
      }
      pageMap={await getPageMap(`/${lang}/docs`)}
      docsRepositoryBase="https://github.com/shuding/nextra/tree/main/docs"
      footer={footer}
      i18n={i18n}
      // ... Your additional layout options
    >
      {children}
    </Layout>
  );
}
