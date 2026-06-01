"""Analise consolidada final da qualificacao (responde aos critiques).
Extrai dos metricas.json: sensibilidade (std entre-sementes E entre-episodios),
OOD, KPIs agronomicos por config, distribuicao de acoes, acuracia BC, testes
estatisticos. Inclui oraculo (com KPIs) e random como referencias.
Salva resumo em results/analise_final.json e imprime tabelas.
"""
import os, sys, json, glob, math
import numpy as np
RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
BASE = os.path.join(RAIZ, 'results', 'experiments')
CONFIGS = ['baseline', 'lr_baixo', 'lr_alto', 'lstm_menor', 'seq_longa']
SEEDS = [42, 123, 2024, 7, 1337]
NOMES = {'baseline':'baseline','lr_baixo':'lr\\_baixo','lr_alto':'lr\\_alto','lstm_menor':'lstm\\_menor','seq_longa':'seq\\_longa'}


def carregar(agente, cfg):
    """Retorna lista de dicts metricas (um por seed)."""
    out = []
    for s in SEEDS:
        f = os.path.join(BASE, agente, 'det', cfg, f'seed_{s}', 'metricas.json')
        if os.path.exists(f):
            out.append(json.load(open(f, encoding='utf-8')))
    return out


def media_std_seeds(metricas, chave_lista):
    """media entre-sementes das medias-por-seed, e std entre-sementes."""
    medias_seed = [np.mean(m[chave_lista]) for m in metricas if m.get(chave_lista)]
    return float(np.mean(medias_seed)), float(np.std(medias_seed)), medias_seed


def std_episodico(metricas, chave_lista):
    """std medio dentro de cada seed (entre os 50 episodios), media entre seeds."""
    stds = [np.std(m[chave_lista]) for m in metricas if m.get(chave_lista)]
    return float(np.mean(stds))


def ic95(valores):
    n = len(valores)
    if n < 2: return (0,0)
    m, s = np.mean(valores), np.std(valores, ddof=1)
    t = 2.776  # t_{0.975, 4} para n=5
    h = t * s / math.sqrt(n)
    return (m - h, m + h)


def teste_pareado(a, b):
    """t pareado entre duas listas (mesmas seeds). Retorna (t, signif_aprox)."""
    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0: return (0, False)
    t = d.mean() / (d.std(ddof=1) / math.sqrt(n))
    # |t| > 2.776 => p<0.05 com df=4
    return (float(t), abs(t) > 2.776)


resumo = {}

print('='*78)
print('  1. SENSIBILIDADE (in-dist): BC e DAgger, std entre-sementes E entre-episodios')
print('='*78)
print(f'{"config":<12} {"BC media":>10} {"BC sd_seed":>10} {"BC sd_ep":>9} | {"DA media":>10} {"DA sd_seed":>10} {"DA sd_ep":>9} {"acur":>6}')
resumo['sensibilidade'] = {}
bc_means, da_means, da_seed_vals = [], [], {}
for c in CONFIGS:
    bc = carregar('il_bc', c); da = carregar('il_dagger', c)
    bcm, bcs, _ = media_std_seeds(bc, 'recompensas_episodios')
    bcep = std_episodico(bc, 'recompensas_episodios')
    dam, das, dav = media_std_seeds(da, 'recompensas_episodios')
    daep = std_episodico(da, 'recompensas_episodios')
    acur = np.mean([m['acuracia_bc_final'] for m in bc])
    bc_means.append(bcm); da_means.append(dam); da_seed_vals[c] = dav
    resumo['sensibilidade'][c] = dict(bc_media=bcm, bc_sd_seed=bcs, bc_sd_ep=bcep,
                                       da_media=dam, da_sd_seed=das, da_sd_ep=daep, acuracia_bc=float(acur))
    print(f'{c:<12} {bcm:>10.0f} {bcs:>10.0f} {bcep:>9.0f} | {dam:>10.0f} {das:>10.0f} {daep:>9.0f} {acur:>6.1%}')
amp_bc = max(bc_means)-min(bc_means); amp_da = max(da_means)-min(da_means)
resumo['amplitude_bc'] = amp_bc; resumo['amplitude_da'] = amp_da
resumo['reducao_pct'] = 100*(1-amp_da/amp_bc)
print(f'\n  Amplitude BC={amp_bc:.0f} | DAgger={amp_da:.0f} | reducao={100*(1-amp_da/amp_bc):.0f}%')

print('\n' + '='*78)
print('  2. TESTE ESTATISTICO (DAgger): melhor vs pior config (t pareado, n=5)')
print('='*78)
melhor = max(CONFIGS, key=lambda c: resumo['sensibilidade'][c]['da_media'])
pior   = min(CONFIGS, key=lambda c: resumo['sensibilidade'][c]['da_media'])
t, sig = teste_pareado(da_seed_vals[melhor], da_seed_vals[pior])
print(f'  Melhor={melhor} ({resumo["sensibilidade"][melhor]["da_media"]:.0f}) vs Pior={pior} ({resumo["sensibilidade"][pior]["da_media"]:.0f})')
print(f'  t pareado = {t:.2f} | significativo (p<0.05)? {"SIM" if sig else "NAO"}')
print('  IC 95% por config (DAgger):')
resumo['ic95'] = {}
for c in CONFIGS:
    lo, hi = ic95(da_seed_vals[c])
    resumo['ic95'][c] = [lo, hi]
    print(f'    {c:<12}: [{lo:.0f}, {hi:.0f}]')
resumo['teste'] = dict(melhor=melhor, pior=pior, t=float(t), significativo=bool(sig))

print('\n' + '='*78)
print('  3. OOD (DAgger) por config + referencias (oraculo, random)')
print('='*78)
resumo['ood'] = {}
for c in CONFIGS:
    da = carregar('il_dagger', c)
    din = np.mean([np.mean(m['recompensas_episodios']) for m in da])
    ddry = np.mean([np.mean(m['ood_dry_recompensas']) for m in da])
    dhum = np.mean([np.mean(m['ood_humid_recompensas']) for m in da])
    resumo['ood'][c] = dict(in_dist=float(din), dry=float(ddry), humid=float(dhum))
    print(f'  {c:<12} in={din:>7.0f} dry={ddry:>7.0f} humid={dhum:>7.0f}')
bot = json.load(open(os.path.join(BASE,'oraculo','avaliacao.json'),encoding='utf-8'))
resumo['oraculo'] = dict(in_dist=bot['in_dist']['media'], dry=bot['ood_dry']['media'], humid=bot['ood_humid']['media'])
print(f'  {"ORACULO":<12} in={bot["in_dist"]["media"]:>7.0f} dry={bot["ood_dry"]["media"]:>7.0f} humid={bot["ood_humid"]["media"]:>7.0f}')
rnd_f = os.path.join(BASE,'random','avaliacao.json')
if os.path.exists(rnd_f):
    rnd = json.load(open(rnd_f,encoding='utf-8'))
    resumo['random'] = dict(in_dist=rnd['in_dist']['media'], dry=rnd['ood_dry']['media'], humid=rnd['ood_humid']['media'])
    print(f'  {"RANDOM":<12} in={rnd["in_dist"]["media"]:>7.0f} dry={rnd["ood_dry"]["media"]:>7.0f} humid={rnd["ood_humid"]["media"]:>7.0f}')

print('\n' + '='*78)
print('  4. KPIs AGRONOMICOS (DAgger) por config')
print('  produtividade | colheitas | agua(regar) | fert(N) | defensivos(herb) | solo')
print('='*78)
resumo['agronomico'] = {}
for c in CONFIGS:
    da = carregar('il_dagger', c)
    prod = np.mean([m['media_produtividade'] for m in da])
    solo = np.mean([m['media_qualidade_solo'] for m in da])
    quim = np.mean([m['media_uso_quimicos'] for m in da])
    # contagem media por episodio (contagem_acoes e total sobre 50 eps)
    def acao_media(da, nome):
        return np.mean([m['contagem_acoes'].get(nome,0)/m['num_episodios'] for m in da])
    colh = acao_media(da,'colher'); agua = acao_media(da,'regar'); fert = acao_media(da,'fertilizar_N'); herb = acao_media(da,'herbicida')
    resumo['agronomico'][c] = dict(produtividade=float(prod), colheitas=float(colh), agua=float(agua),
                                    fertilizante=float(fert), defensivos=float(herb), uso_quimicos=float(quim), solo=float(solo))
    print(f'  {c:<12} {prod:>8.0f} {colh:>10.1f} {agua:>10.1f} {fert:>8.1f} {herb:>13.1f} {solo:>7.2f}')

print('\n' + '='*78)
print('  5. DISTRIBUICAO DE ACOES (DAgger baseline vs Oraculo nas demos)')
print('='*78)
# Oraculo: contar acoes nas demos (trajetorias_especialista.json pode estar ausente)
_dfiles = sorted(glob.glob(os.path.join(RAIZ,'results','bot_perfeito','**','trajetorias_especialista.json'),recursive=True))
if not _dfiles:
    print('  (trajetorias_especialista.json ausente; pulando a distribuicao de acoes.')
    print('   Gere as demonstracoes com bot_perfeito.py para recalcular esta secao.)')
else:
    demos = json.load(open(_dfiles[-1],encoding='utf-8'))
    NOMES_ACOES = ['plantar','colher','regar','fertilizar_N','fertilizar_P','fertilizar_K','fertilizar_C','herbicida','pesticida','espantalho_basico','espantalho_avancado','remover_espantalho','colocar_cerca','observar','esperar']
    cont_oraculo = {}
    tot_o = 0
    for ep in demos:
        for t in ep:
            a = NOMES_ACOES[t['acao']]; cont_oraculo[a]=cont_oraculo.get(a,0)+1; tot_o+=1
    da_base = carregar('il_dagger','baseline')
    cont_ag = {}; tot_a = 0
    for m in da_base:
        for k,v in m['contagem_acoes'].items(): cont_ag[k]=cont_ag.get(k,0)+v; tot_a+=v
    print(f'{"acao":<14} {"oraculo%":>9} {"agente%":>9}')
    resumo['acoes'] = {}
    for a in sorted(set(list(cont_oraculo)+list(cont_ag)), key=lambda x:-(cont_oraculo.get(x,0))):
        po = 100*cont_oraculo.get(a,0)/tot_o; pa = 100*cont_ag.get(a,0)/tot_a
        if po>0.5 or pa>0.5:
            resumo['acoes'][a] = dict(oraculo=po, agente=pa)
            print(f'{a:<14} {po:>8.1f}% {pa:>8.1f}%')

# BC-longo (se existir)
print('\n' + '='*78)
print('  6. CONTROLE BC-LONGO (170 epocas) vs BC(100) vs DAgger')
print('='*78)
if os.path.exists(os.path.join(BASE,'il_bc_longo')):
    resumo['bc_longo'] = {}
    print(f'{"config":<12} {"BC-100":>8} {"BC-170":>8} {"DAgger":>8}')
    for c in CONFIGS:
        bcl = carregar('il_bc_longo', c)
        if not bcl: continue
        bclm = np.mean([np.mean(m['recompensas_episodios']) for m in bcl])
        resumo['bc_longo'][c] = float(bclm)
        print(f'  {c:<12} {resumo["sensibilidade"][c]["bc_media"]:>8.0f} {bclm:>8.0f} {resumo["sensibilidade"][c]["da_media"]:>8.0f}')
else:
    print('  (BC-longo ainda nao concluido)')

json.dump(resumo, open(os.path.join(RAIZ,'results','analise_final.json'),'w',encoding='utf-8'), indent=2, ensure_ascii=False)
print('\n[salvo] results/analise_final.json')
