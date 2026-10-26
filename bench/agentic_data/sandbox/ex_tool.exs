# The Elixir side of the agentic data pipeline (workspace.py, tasks.py, agent.py): reads one source file with Elixir's
# own parser and prints one JSON object. Runs inside the Elixir sandbox (Dockerfile.elixir); standard library only.
#
#   elixir ex_tool.exs modules FILE    the modules the file defines at its top level, and every alias that names one of
#                                      them or a module inside them (line, column, text): the places to rename
#   elixir ex_tool.exs mutations FILE  single-token changes inside def and defp clauses: comparison, arithmetic and
#                                      boolean operators, the integers 0 to 2, true/false, a dropped not or !
#   elixir ex_tool.exs stubs FILE      documented single-clause public functions with a do-block body of 3+ lines
#   elixir ex_tool.exs docs FILE       sha256 of the file's @moduledoc, @doc and @typedoc values, in order
#
# Lines and columns count from 1; a column counts characters, as Elixir's tokenizer does.

defmodule ExTool do
  @flip %{
    :< => {:<=, "comparison"}, :<= => {:<, "comparison"}, :> => {:>=, "comparison"}, :>= => {:>, "comparison"},
    :== => {:!=, "comparison"}, :!= => {:==, "comparison"}, :=== => {:!==, "comparison"},
    :!== => {:===, "comparison"}, :+ => {:-, "arithmetic"}, :- => {:+, "arithmetic"},
    :and => {:or, "boolean"}, :or => {:and, "boolean"}, :&& => {:||, "boolean"}, :|| => {:&&, "boolean"}
  }

  def main([cmd, file]) do
    src = File.read!(file)

    out =
      case cmd do
        "modules" -> modules(src, file)
        "mutations" -> mutations(src, file)
        "stubs" -> stubs(src, file)
        "docs" -> docs(src, file)
      end

    IO.puts(JSON.encode!(out))
  end

  defp parse(src, file, opts \\ []) do
    Code.string_to_quoted!(src, [file: file, columns: true, token_metadata: true, emit_warnings: false] ++ opts)
  end

  defp name(segs), do: Enum.map_join(segs, ".", &Atom.to_string/1)

  # ---- modules: what to rename

  def modules(src, file) do
    ast = parse(src, file)
    tops = top_modules(ast)

    {_, refs} =
      Macro.prewalk(ast, [], fn
        {:__aliases__, meta, segs} = node, acc when is_list(segs) ->
          if Enum.all?(segs, &is_atom/1) and meta[:line] != nil and
               Enum.any?(tops, &List.starts_with?(segs, &1)) do
            {node, [%{line: meta[:line], column: meta[:column], text: name(segs)} | acc]}
          else
            {node, acc}
          end

        node, acc ->
          {node, acc}
      end)

    %{modules: Enum.map(tops, &name/1), refs: Enum.reverse(refs)}
  end

  defp top_modules({:__block__, _, exprs}), do: Enum.flat_map(exprs, &top_modules/1)

  defp top_modules({:defmodule, _, [{:__aliases__, _, segs}, _]}) when is_list(segs) do
    if Enum.all?(segs, &is_atom/1), do: [segs], else: []
  end

  defp top_modules(_), do: []

  # ---- every def/defp clause: module, name, arity, lines, documented

  defp clauses(ast), do: elem(walk(ast, [], {[], nil}), 0) |> Enum.reverse()

  defp walk({:defmodule, _, [{:__aliases__, _, segs}, [{_, body}]]}, mods, {acc, _doc}) when is_list(segs) do
    {acc, _} = walk(body, mods ++ segs, {acc, nil})
    {acc, nil}
  end

  defp walk({:__block__, _, exprs}, mods, state), do: Enum.reduce(exprs, state, &walk(&1, mods, &2))

  defp walk({:@, _, [{:doc, _, [value]}]}, _mods, {acc, _doc}), do: {acc, value != false}

  defp walk({kind, meta, [head | _]} = node, mods, {acc, doc}) when kind in [:def, :defp] do
    {fname, arity} = name_arity(head)

    clause = %{
      module: name(mods), name: fname, arity: arity, public: kind == :def, documented: doc == true,
      line: meta[:line], last: last_line(node, meta), do: meta[:do][:line], end: meta[:end][:line], node: node
    }

    {[clause | acc], nil}
  end

  defp walk(_, _mods, {acc, doc}), do: {acc, doc}

  defp name_arity({:when, _, [call | _]}), do: name_arity(call)
  defp name_arity({fname, _, args}) when is_atom(fname), do: {Atom.to_string(fname), if(is_list(args), do: length(args), else: 0)}
  defp name_arity(_), do: {"?", 0}

  defp last_line(node, meta) do
    case meta[:end] do
      nil ->
        {_, max} =
          Macro.prewalk(node, 0, fn
            {_, m, _} = n, acc when is_list(m) -> {n, max(acc, m[:line] || 0)}
            n, acc -> {n, acc}
          end)

        max

      e ->
        e[:line]
    end
  end

  defp label(c), do: "#{c.module}.#{c.name}/#{c.arity}"

  # ---- mutations

  def mutations(src, file) do
    ast = parse(src, file)
    cl = clauses(ast)

    ops =
      Enum.flat_map(cl, fn c ->
        {_, found} =
          Macro.prewalk(c.node, [], fn
            {op, meta, [_, _]} = n, acc when is_map_key(@flip, op) ->
              {new, kind} = @flip[op]
              {n, [mut(meta, op, new, kind, c) | acc]}

            {op, meta, [_]} = n, acc when op in [:not, :!] ->
              {n, [mut(meta, op, "", "dropped not", c) | acc]}

            n, acc ->
              {n, acc}
          end)

        found
      end)

    lit = parse(src, file, literal_encoder: &{:ok, {:__block__, &2, [&1]}})

    {_, lits} =
      Macro.prewalk(lit, [], fn
        {:__block__, meta, [v]} = n, acc when (is_integer(v) and v in 0..2) or is_boolean(v) ->
          case Enum.find(cl, &(meta[:line] != nil and &1.line <= meta[:line] and meta[:line] <= &1.last)) do
            nil ->
              {n, acc}

            c ->
              old = if is_boolean(v), do: to_string(v), else: meta[:token] || Integer.to_string(v)
              new = if is_boolean(v), do: to_string(!v), else: %{0 => "1", 1 => "2", 2 => "1"}[v]
              {n, [mut(meta, old, new, "constant", c) | acc]}
          end

        n, acc ->
          {n, acc}
      end)

    %{mutations: Enum.filter(ops ++ lits, &(&1.line != nil)) |> Enum.sort_by(&{&1.line, &1.column})}
  end

  defp mut(meta, old, new, kind, c) do
    %{line: meta[:line], column: meta[:column], old: to_string(old), new: to_string(new), kind: kind, function: label(c)}
  end

  # ---- stubs

  def stubs(src, file) do
    cl = clauses(parse(src, file))
    counts = Enum.frequencies_by(cl, &label/1)

    found =
      for c <- cl, c.public, c.documented, counts[label(c)] == 1, c.do != nil, c.end != nil,
          c.end - c.do - 1 >= 3 do
        %{function: label(c), first: c.do + 1, last: c.end - 1}
      end

    %{stubs: found}
  end

  # ---- docs

  def docs(src, file) do
    {_, acc} =
      Macro.prewalk(parse(src, file), [], fn
        {:@, _, [{attr, _, [v]}]} = n, acc when attr in [:moduledoc, :doc, :typedoc] -> {n, [{attr, strip(v)} | acc]}
        n, acc -> {n, acc}
      end)

    bin = :erlang.term_to_binary(Enum.reverse(acc))
    %{count: length(acc), sha256: Base.encode16(:crypto.hash(:sha256, bin), case: :lower)}
  end

  defp strip(v), do: Macro.prewalk(v, fn {a, m, b} when is_list(m) -> {a, [], b}; x -> x end)
end

ExTool.main(System.argv())
