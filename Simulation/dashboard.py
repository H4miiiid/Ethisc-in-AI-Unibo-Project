from datetime import timedelta
import streamlit as st
from streamlit_option_menu import option_menu

from areaverde_simulation import *

if '__model__' not in st.session_state:
    m = Model()
    st.session_state['__model__'] = m
else:
    m = st.session_state['__model__']

params = [
    {"name": "Costi",
     "params": [
         {'id': m.I_P_cost[e], 'label': f"Costo ingresso Euro {e} (in €)", 'value': m.I_P_cost[e].value,
          'min': 0.0, 'max': 10.0, 'step': 0.25} for e in range(7)
     ]},
    {"name": "Parametri regolamento",
     "params": [
         {'id': m.I_P_start_time, 'label': "Ora inizio", 'value': to_time(m.I_P_start_time.value),
          'min': to_time(0), 'max': to_time(3600 * 24 - 900), 'step': timedelta(seconds=900), 'type': 'Time'},
         {'id': m.I_P_end_time, 'label': "Ora termine", 'value': to_time(m.I_P_end_time.value),
          'min': to_time(0), 'max': to_time(3600 * 24 - 900), 'step': timedelta(seconds=900), 'type': 'Time'},
         {'id': m.I_P_fraction_exempted, 'label': "% veicoli esonerati", 'value': m.I_P_fraction_exempted.value,
          'min': 0.0, 'max': 1.0, 'step': 0.05},
     ]},
    {"name": "Parametri comportamentali",
     "params": [
         {'id': m.I_B_p50_cost, 'label': "Soglia di costo accettabile (in €, mediano)",
          'value': (m.I_B_p50_cost.loc, m.I_B_p50_cost.scale),
          'min': 0.0, 'max': 10.0, 'step': 0.25},
         {'id': m.I_B_p50_anticipating, 'label': "Disponibilità anticipo viaggio (in ore, mediano)",
          'value': m.I_B_p50_anticipating.value,
          'min': 0.10, 'max': 6.0, 'step': 0.10},
         {'id': m.I_B_p50_postponing, 'label': "Disponibilità posticipo viaggio (in ore, mediano)",
          'value': m.I_B_p50_postponing.value,
          'min': 0.10, 'max': 6.0, 'step': 0.10}] 
          + 
          ([] if TIME_SHIFT_STRATEGY == "fixed" else [
         {'id': m.I_B_p50_anticipation, 'label': "Tempo mediano di anticipo (in ore)",
            'value': m.I_B_p50_anticipation.value,
            'min': 0.10, 'max': 6.0, 'step': 0.10},
         {'id': m.I_B_p50_postponement, 'label': "Tempo mediano mediano posticipo (in ore)",
            'value': m.I_B_p50_postponement.value,#TODO: minimo qui = valore di posticipating
            'min': 0.10, 'max': 6.0, 'step': 0.10} ])
            + 
          ([] if MODAL_SHIFT_OPTION == "no" or MODAL_SHIFT_OPTION == "active" else [
         {'id': m.I_B_pt_comfort, 'label': "Livello di importanza del comfort del TPM",
          'value': m.I_B_pt_comfort.value,
          'min': 0.0, 'max': 5.0, 'step': 0.1},
         {'id': m.I_B_pt_capillarity, 'label': "Livello di importanza della capillatà del TPM",
          'value': m.I_B_pt_capillarity.value,
          'min': 0.0, 'max': 5.0, 'step': 0.1},
         {'id': m.I_B_pt_frequency, 'label': "Livello di importanza della frequenza del TPM",
          'value': m.I_B_pt_frequency.value,
          'min': 0.0, 'max': 5.0, 'step': 0.1},
         {'id': m.I_B_pt_cost, 'label': "Livello di importanza del costo del TPM",
          'value': m.I_B_pt_cost.value,
          'min': 0.0, 'max': 5.0, 'step': 0.1} ])
           + [
         {'id': m.I_B_starting_modified_factor, 'label': "Modifica circolazione in Area Verde",
          'value': m.I_B_starting_modified_factor.value,
          'min': 0.0, 'max': 2.0, 'step': 0.10}]
     },
]

all_params = [p for g in params for p in g["params"]]

#################################################################################
#####################          STREAMLIT APP              #######################
#################################################################################
st.set_page_config(
    page_title="Area Verde",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    selected = option_menu(
        menu_title="Area Verde",
        options=["Home"],
        icons=["house"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    c = {}

    ZONE = 0
    ZONE_NAME = ""
    TIME = (to_time(0), to_time(3600 * 24))
    DELTA = False

    # Create a row layout
    c_plot, c_form = st.columns([2, 1])

    # Parameters
    with c_form:
        reset = st.button("Reset", type="primary")
        if reset:
            for p in all_params:
                p['value'] = st.session_state[p['id'].name + "_default"]
                st.session_state[p['id'].name] = p['value']

        c_form.header("Parametri")
        for g in params:
            with st.expander(g["name"], False):
                for p in g["params"]:
                    if p['id'].name not in st.session_state:
                        st.session_state[p['id'].name + "_default"] = p['value']
                        st.session_state[p['id'].name] = p['value']
                    time_format = 'HH:mm' if 'type' in p.keys() and p['type'] == 'Time' else None
                    c[p['id'].name] = st.slider(p['label'], min_value=p['min'], max_value=p['max'], step=p['step'],
                                                format=time_format, key=p['id'].name)

        with st.expander("Visualizzazione", True):
            viewtype = st.radio("Tipo grafico", ["Area Verde", "Zona", "Mappa", "Mappa Aree Statistiche"]) # Add new section
            if viewtype == "Zona":
                multi_zone = st.checkbox("Zone multiple")
                if multi_zone:
                    zone = st.multiselect("Zone", sorted(names_to_zones), placeholder="Tutte le zone")
                else:
                    zone = st.selectbox("Zona", sorted(names_to_zones))
            if viewtype == "Mappa":
                time_interval = st.checkbox("Intervallo")
                if time_interval:
                    time = st.slider("Orario", value=TIME,
                                     min_value=to_time(0), max_value=to_time(3600 * 24), step=timedelta(seconds=900),
                                     format="HH:mm")
                else:
                    time = st.slider("Orario", value=to_time(3600 * 6),
                                     min_value=to_time(0), max_value=to_time(3600 * 24), step=timedelta(seconds=900),
                                     format="HH:mm")
                delta = st.checkbox("Mostra differenze")
            # The conditions in the new section
            if viewtype == "Mappa Aree Statistiche":
                time_interval = st.checkbox("Intervallo")
                if time_interval:
                    time = st.slider("Orario", value=TIME,
                                     min_value=to_time(0), max_value=to_time(3600 * 24), step=timedelta(seconds=900),
                                     format="HH:mm")
                else:
                    time = st.slider("Orario", value=to_time(3600 * 6),
                                     min_value=to_time(0), max_value=to_time(3600 * 24), step=timedelta(seconds=900),
                                     format="HH:mm")
                delta = st.checkbox("Mostra differenze")

                visualization_mode = st.selectbox(
                    "Modalità di ponderazione etica",
                    options=["raw", "fragility_weighted", "gender_weighted", "ethics_weighted"],
                    format_func=lambda x: {
                        "raw": "Normale (non pesato)",
                        "fragility_weighted": "Pesato per fragilità",
                        "gender_weighted": "Pesato per % donne",
                        "ethics_weighted": "Pesato per entrambi (fragilità + % donne)"
                    }[x],
                    index=0
                )
            match viewtype:
                case "Area Verde":
                    ZONE = 0
                case "Zona":
                    if not zone:
                        ZONE = zones
                        ZONE_NAME = sorted(names_to_zones)
                    elif type(zone) is list:
                        ZONE = [names_to_zones[z] for z in zone]
                        ZONE_NAME = zone
                    else:
                        ZONE = [names_to_zones[zone]]
                        ZONE_NAME = [zone]
                case "Mappa":
                    ZONE = -1
                    TIME = time
                    DELTA = delta
                case "Mappa Aree Statistiche":
                    ZONE = -2
                    TIME = time
                    DELTA = delta
                case _:
                    ZONE = 0

    with (c_plot):
        changed = False
        for p in all_params:
            if 'type' in p.keys() and p['type'] == 'Time':
                if isinstance(p['id'], UniformDistIndex):
                    (l, s) = st.session_state[p['id'].name]
                    if (p['id'].loc, p['id'].scale) != (to_number(l), to_number(s)):
                        (p['id'].loc, p['id'].scale) = (to_number(l), to_number(s))
                        changed = True
                else:
                    if p['id'].value != to_number(st.session_state[p['id'].name]):
                        p['id'].value = to_number(st.session_state[p['id'].name])
                        changed = True
            else:
                if isinstance(p['id'], UniformDistIndex):
                    if (p['id'].loc, p['id'].scale) != st.session_state[p['id'].name]:
                        (p['id'].loc, p['id'].scale) = st.session_state[p['id'].name]
                        changed = True
                else:
                    if p['id'].value != st.session_state[p['id'].name]:
                        p['id'].value = st.session_state[p['id'].name]
                        changed = True

        if '__subs__' not in st.session_state or changed:
            subs = m.evaluate(20)
            st.session_state['__subs__'] = subs
        else:
            subs = st.session_state['__subs__']

        st.subheader("Veicoli in ingresso")
        if ZONE == 0:
            fig = plot_field_graph(subs[m.I_modified_inflow],
                                   horizontal_label="Ora", vertical_label="Flusso in ingresso [veicoli/ora]",
                                   vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                                   reference_line=subs[m.TS_inflow][0])
            st.pyplot(fig, use_container_width=False, clear_figure=True)
        elif ZONE == -1:
            if DELTA:
                fig = plot_map_graph(subs, m.I_delta_inflow_zone, time=TIME,
                                     label="Differenza flusso in ingresso [veicoli/ora]")
            else:
                fig = plot_map_graph(subs, m.I_modified_inflow_zone, time=TIME, range=(0, None), label="Flusso")
            st.plotly_chart(fig, use_container_width=False)
        elif ZONE == -2:
            if DELTA:
                fig = plot_statistical_area_map(subs, m.I_delta_inflow_zone, time=TIME,
                                                label="Differenza flusso in ingresso [veicoli/ora]", visualization_mode= visualization_mode)
            else:
                fig = plot_statistical_area_map(subs, m.I_modified_inflow_zone, time=TIME, range_val=(0, None), label="Flusso", visualization_mode= visualization_mode)
            st.plotly_chart(fig, use_container_width=False, config={"scrollZoom": True})
        else:
            fig = plot_multifield_graph(subs, m.I_modified_inflow_zone, ZONE,
                                        horizontal_label="Ora", vertical_label="Flusso in ingresso [veicoli/ora]",
                                        vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                                        reference_index=m.TS_inflow_zone)
            st.pyplot(fig, use_container_width=False, clear_figure=True)

        st.subheader("Traffico")
        if ZONE == 0:
            fig = plot_field_graph(subs[m.I_modified_traffic],
                                   horizontal_label="Ora", vertical_label="Traffico [veicoli circolanti]",
                                   reference_line=subs[m.I_traffic][0])
            st.pyplot(fig, use_container_width=False, clear_figure=True)
        elif ZONE == -1:
            if DELTA:
                fig = plot_map_graph(subs, m.I_delta_traffic_zone, time=TIME,
                                     label="Differenza traffico [delta media veicoli circolanti]",
                                     function=statistics.mean)
            else:
                fig = plot_map_graph(subs, m.I_modified_traffic_zone, time=TIME,
                                     label="Traffico [max veicoli circolanti]", range=(0, None), function=max)
            st.plotly_chart(fig, use_container_width=False)
        elif ZONE == -2:
            if DELTA:
                fig = plot_statistical_area_map(subs, m.I_delta_traffic_zone, time=TIME,
                                     label="Differenza traffico [delta media veicoli circolanti]",
                                     function=statistics.mean, visualization_mode= visualization_mode)
            else:
                fig = plot_statistical_area_map(subs, m.I_modified_traffic_zone, time=TIME,
                                     label="Traffico [max veicoli circolanti]", range_val=(0, None), function=max, visualization_mode= visualization_mode)
            st.plotly_chart(fig, use_container_width= False, config={"scrollZoom": True})
        else:
            fig = plot_multifield_graph(subs, m.I_modified_traffic_zone, ZONE,
                                        horizontal_label="Ora", vertical_label="Traffico [veicoli circolanti]",
                                        reference_index=m.TS_traffic_zone)
            st.pyplot(fig, use_container_width=False, clear_figure=True)

        st.subheader("Emissioni")
        if ZONE == 0:
            fig = plot_field_graph(subs[m.I_modified_emissions],
                                   horizontal_label="Ora", vertical_label="Emissioni [NOx gr/ora]",
                                   vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                                   reference_line=subs[m.I_emissions][0])
            st.pyplot(fig, use_container_width=False, clear_figure=True)
        elif ZONE == -1:
            if DELTA:
                fig = plot_map_graph(subs, m.I_delta_emissions_zone, time=TIME,
                                     label="Differenza emissioni [NOx gr/giorno]")
            else:
                fig = plot_map_graph(subs, m.I_modified_emissions_zone, time=TIME,
                                     label="Emissioni [NOx gr/giorno]", range=(0, None))
            st.plotly_chart(fig, use_container_width=False)
        elif ZONE == -2:
            if DELTA:
                fig = plot_statistical_area_map(subs, m.I_delta_emissions_zone, time=TIME,
                                     label="Differenza emissioni [NOx gr/giorno]", visualization_mode= visualization_mode)
            else:
                fig = plot_statistical_area_map(subs, m.I_modified_emissions_zone, time=TIME,
                                     label="Emissioni [NOx gr/giorno]", range_val=(0, None), visualization_mode= visualization_mode)
            st.plotly_chart(fig, use_container_width= False, config={"scrollZoom": True})
        else:
            fig = plot_multifield_graph(subs, m.I_modified_emissions_zone, ZONE,
                                        horizontal_label="Ora", vertical_label="Emissioni [NOx gr/ora]",
                                        vertical_formatter=FuncFormatter(lambda x, _: f"{int(x * 12)}"),
                                        reference_index=m.I_emissions_zone)
            st.pyplot(fig, use_container_width=False, clear_figure=True)

        kpi_translation = {
            'Base inflow [veh/day]': 'Flusso attuale (in veicoli/giorno)',
            'Modified inflow [veh/day]': 'Flusso ridotto (in veicoli/giorno)',
            'Time-shifted inflow [veh/day]': 'Flusso anticipato/posticipato (in veicoli/giorno)',
            'Mode-shifted inflow [veh/day]': 'Flusso trasferito su TPM (in veicoli/giorno)',
            'Lost inflow [veh/day]': 'Flusso perso (in veicoli/giorno)',
            'Paying inflow [veh/day]': 'Veicoli paganti (in veicoli/giorno)',
            'Collected fees [€/day]': 'Pagamenti collezionati (in €/giorno)',
            'Emissions [NOx gr/day]': "Emissioni (in NOx gr/giorno)",
            'Emissions difference [NOx gr/day]': 'Differenza in emissioni (in NOx gr/giorno)'
        }

        if ZONE == 0 or ZONE == -1 or ZONE == -2:
            st.subheader("Indicatori (giornalieri)")
            for k, v in compute_kpis(m, subs).items():
                if k in kpi_translation:
                    k = kpi_translation[k]
                st.write(f'{k} - {v:_}'.replace('_', '.'))

        time_index = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]


        def prepare_dataframe(index, zone_index):
            df = pd.DataFrame(subs[index].mean(axis=0), index=time_index, columns=["Area Verde"])
            for z in zones:
                zone_name = zones_to_names[z]
                df[zone_name] = subs[zone_index[z]].mean(axis=0)
            return df


        st.subheader("Scaricamento dati")
        st.download_button('Veicoli in ingresso (riferimento)',
                           prepare_dataframe(m.TS_inflow, m.TS_inflow_zone).to_csv().encode('utf-8'),
                           file_name="veicoli in ingresso rif.csv", mime='text/csv')
        st.download_button('Veicoli in ingresso (modificato)',
                           prepare_dataframe(m.I_modified_inflow, m.I_modified_inflow_zone).to_csv().encode('utf-8'),
                           file_name="veicoli in ingresso modificato.csv", mime='text/csv')
        st.download_button('Traffico (riferimento)',
                           prepare_dataframe(m.I_traffic, m.TS_traffic_zone).to_csv().encode('utf-8'),
                           file_name="traffico rif.csv", mime='text/csv')
        st.download_button('Traffico (modificato)',
                           prepare_dataframe(m.I_modified_traffic, m.I_modified_traffic_zone).to_csv().encode('utf-8'),
                           file_name="traffico modificato.csv", mime='text/csv')
        st.download_button('Emissioni (riferimento)',
                           prepare_dataframe(m.I_emissions, m.I_emissions_zone).to_csv().encode('utf-8'),
                           file_name="emissioni rif.csv", mime='text/csv')
        st.download_button('Emissioni (modificato)',
                           prepare_dataframe(m.I_modified_emissions, m.I_modified_emissions_zone).
                           to_csv().encode('utf-8'),
                           file_name="emissioni modificato.csv", mime='text/csv')