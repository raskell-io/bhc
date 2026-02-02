//! Pretty-printing for Core IR.
//!
//! Renders Core IR as readable Haskell-like syntax, suitable for
//! inspection and debugging. Uses standard `fmt::Display` trait.

use std::fmt;

use crate::{Alt, AltCon, Bind, Coercion, CoreModule, DataCon, Expr, Literal, Var};

impl fmt::Display for CoreModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "module {} where", self.name)?;
        for bind in &self.bindings {
            writeln!(f)?;
            write!(f, "{bind}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Bind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bind::NonRec(var, expr) => {
                write!(f, "{} = {}", var.name, expr)
            }
            Bind::Rec(binds) => {
                for (i, (var, expr)) in binds.iter().enumerate() {
                    if i > 0 {
                        writeln!(f)?;
                    }
                    write!(f, "{} = {}", var.name, expr)?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write_expr(f, self, 0)
    }
}

/// Write an expression with a precedence context.
///
/// Precedence levels:
/// - 0: top-level (no parens needed)
/// - 1: lambda body, let body, case body
/// - 2: function application (left-associative)
/// - 3: atomic (variables, literals — never need parens)
fn write_expr(f: &mut fmt::Formatter<'_>, expr: &Expr, prec: u8) -> fmt::Result {
    match expr {
        Expr::Var(var, _) => write!(f, "{}", var.name),

        Expr::Lit(lit, _, _) => write!(f, "{lit}"),

        Expr::App(func, arg, _) => {
            let needs_parens = prec > 2;
            if needs_parens {
                write!(f, "(")?;
            }
            write_expr(f, func, 2)?;
            write!(f, " ")?;
            write_expr(f, arg, 3)?;
            if needs_parens {
                write!(f, ")")?;
            }
            Ok(())
        }

        Expr::TyApp(inner, _ty, _) => {
            // Type applications are erased in pretty output
            write_expr(f, inner, prec)
        }

        Expr::Lam(var, body, _) => {
            let needs_parens = prec > 0;
            if needs_parens {
                write!(f, "(")?;
            }
            write!(f, "\\{} -> ", var.name)?;
            write_expr(f, body, 0)?;
            if needs_parens {
                write!(f, ")")?;
            }
            Ok(())
        }

        Expr::TyLam(_, body, _) => {
            // Type lambdas are erased
            write_expr(f, body, prec)
        }

        Expr::Let(bind, body, _) => {
            let needs_parens = prec > 0;
            if needs_parens {
                write!(f, "(")?;
            }
            match bind.as_ref() {
                Bind::NonRec(var, rhs) => {
                    write!(f, "let {} = ", var.name)?;
                    write_expr(f, rhs, 0)?;
                    write!(f, " in ")?;
                }
                Bind::Rec(binds) => {
                    write!(f, "let ")?;
                    for (i, (var, rhs)) in binds.iter().enumerate() {
                        if i > 0 {
                            write!(f, "; ")?;
                        }
                        write!(f, "{} = ", var.name)?;
                        write_expr(f, rhs, 0)?;
                    }
                    write!(f, " in ")?;
                }
            }
            write_expr(f, body, 0)?;
            if needs_parens {
                write!(f, ")")?;
            }
            Ok(())
        }

        Expr::Case(scrut, alts, _, _) => {
            let needs_parens = prec > 0;
            if needs_parens {
                write!(f, "(")?;
            }
            write!(f, "case ")?;
            write_expr(f, scrut, 0)?;
            write!(f, " of {{ ")?;
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    write!(f, "; ")?;
                }
                write!(f, "{alt}")?;
            }
            write!(f, " }}")?;
            if needs_parens {
                write!(f, ")")?;
            }
            Ok(())
        }

        Expr::Lazy(inner, _) => {
            write!(f, "lazy ")?;
            write_expr(f, inner, 3)
        }

        Expr::Cast(inner, _, _) | Expr::Tick(_, inner, _) => write_expr(f, inner, prec),

        Expr::Type(ty, _) => write!(f, "@{ty:?}"),

        Expr::Coercion(_, _) => write!(f, "<coercion>"),
    }
}

impl fmt::Display for Alt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.con)?;
        for binder in &self.binders {
            write!(f, " {}", binder.name)?;
        }
        write!(f, " -> ")?;
        write_expr(f, &self.rhs, 0)
    }
}

impl fmt::Display for AltCon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AltCon::DataCon(dc) => write!(f, "{}", dc.name),
            AltCon::Lit(lit) => write!(f, "{lit}"),
            AltCon::Default => write!(f, "_"),
        }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Int(n) => write!(f, "{n}"),
            Literal::Integer(n) => write!(f, "{n}"),
            Literal::Float(v) => write!(f, "{v}"),
            Literal::Double(v) => write!(f, "{v}"),
            Literal::Char(c) => write!(f, "'{c}'"),
            Literal::String(s) => write!(f, "\"{}\"", s.as_str()),
        }
    }
}

impl fmt::Display for Var {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl fmt::Display for DataCon {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bhc_index::Idx;
    use bhc_intern::Symbol;
    use bhc_span::Span;
    use bhc_types::Ty;

    use crate::VarId;

    fn mk_var(name: &str) -> Var {
        Var::new(Symbol::intern(name), VarId::new(0), Ty::Error)
    }

    #[test]
    fn test_display_literal() {
        let lit = Expr::Lit(Literal::Int(42), Ty::Error, Span::default());
        assert_eq!(format!("{lit}"), "42");
    }

    #[test]
    fn test_display_var() {
        let var = mk_var("x");
        let expr = Expr::Var(var, Span::default());
        assert_eq!(format!("{expr}"), "x");
    }

    #[test]
    fn test_display_app() {
        let f = mk_var("f");
        let x = mk_var("x");
        let expr = Expr::App(
            Box::new(Expr::Var(f, Span::default())),
            Box::new(Expr::Var(x, Span::default())),
            Span::default(),
        );
        assert_eq!(format!("{expr}"), "f x");
    }

    #[test]
    fn test_display_lam() {
        let x = mk_var("x");
        let body = Expr::Var(x.clone(), Span::default());
        let expr = Expr::Lam(x, Box::new(body), Span::default());
        assert_eq!(format!("{expr}"), "\\x -> x");
    }

    #[test]
    fn test_display_let() {
        let x = mk_var("x");
        let bind = Bind::NonRec(x.clone(), Box::new(Expr::Lit(Literal::Int(1), Ty::Error, Span::default())));
        let body = Expr::Var(x, Span::default());
        let expr = Expr::Let(Box::new(bind), Box::new(body), Span::default());
        assert_eq!(format!("{expr}"), "let x = 1 in x");
    }

    #[test]
    fn test_display_string_literal() {
        let s = Symbol::intern("hello");
        let lit = Expr::Lit(Literal::String(s), Ty::Error, Span::default());
        assert_eq!(format!("{lit}"), "\"hello\"");
    }
}
