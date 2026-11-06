# RailPulse — Note de cadrage du projet personnel
## Question métier
Anticiper les tendances mensuelles du taux de retard par liaison ferroviaire pour mieux cibler l'investigation opérationnelle.

## Données
Source publique SNCF Voyageurs : régularité Intercités par mois, gare de départ et gare d'arrivée (ODbL). Aucune donnée ONCF utilisée. Les seuils de retard varient avec la durée du trajet; comparer directement aux KPI ONCF serait incorrect.

## Méthodologie
1. Télécharger le CSV officiel, contrôler les schémas et les contraintes métier.
2. Calculer le taux de retard mensuel par liaison et le taux d'annulation.
3. Préparer les variables temporelles décalées d'un à trois mois, puis la saisonnalité.
4. Séparer les six derniers mois pour une évaluation chronologique.
5. Comparer Random Forest et prévision naïve fondée sur le mois précédent.
6. Visualiser les erreurs et les prévisions du mois suivant sur Streamlit.

## KPI
MAE et RMSE en points de pourcentage; nombre de liaisons représentées; période d'évaluation; écart de performance du modèle vs baseline.

## Limitations
- Une prévision de taux mensuel n'est pas une prévision de retard individuel.
- Pas de preuve de généralisation au réseau ONCF.
- Résultats de tests synthétiques non assimilables aux performances sur SNCF réel.
- Pas de météo, pas de données d'exploitation en temps réel.
- La régularité composite SNCF est définie selon la durée du trajet.
